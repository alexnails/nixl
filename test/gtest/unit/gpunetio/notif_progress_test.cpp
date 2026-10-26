/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gpunetio_backend_aux.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <endian.h>
#include <vector>

namespace {

// RQ CQ entries of a fake notification QP; no peer gets near a full lap of them.
constexpr uint32_t rq_cqe_num = 64;
constexpr auto mailbox_timeout = std::chrono::seconds(5);

static_assert(sizeof(doca_gpu_dev_verbs_cqe64) == DOCA_GPUNETIO_VERBS_CQE_SIZE);

// Runs the plugin's progress kernel against fake notification QPs. The test writes receive CQEs
// in place of the NIC and polls the notification mailbox the way getNotifs() does.
class GpunetioNotifProgress : public ::testing::Test {
protected:
    struct Peer {
        doca_gpu_dev_verbs_qp *qp = nullptr;
        uint8_t *cqes = nullptr;
        __be32 *dbrec = nullptr;
        uint64_t delivered = 0;
        uint64_t received = 0;
    };

    void
    SetUp() override {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "No CUDA device available";
        }
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        ASSERT_EQ(cudaStreamCreateWithFlags(&progress_stream_, cudaStreamNonBlocking), cudaSuccess);
        ASSERT_EQ(cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&ring_, sizeof(docaSlotDesc) * DOCA_XFER_REQ_MAX), cudaSuccess);
        ASSERT_EQ(cudaMemset(ring_, 0, sizeof(docaSlotDesc) * DOCA_XFER_REQ_MAX), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&gpu_state_, sizeof(docaGpuState)), cudaSuccess);
        ASSERT_EQ(cudaMemset(gpu_state_, 0, sizeof(docaGpuState)), cudaSuccess);
        // As in the engine, the host state and the mailboxes live in mapped host memory.
        ASSERT_NO_FATAL_FAILURE(allocMapped(host_state_, host_state_gpu_));
        ASSERT_NO_FATAL_FAILURE(allocMapped(notif_fill_, notif_fill_gpu_));
        ASSERT_NO_FATAL_FAILURE(allocMapped(notif_progress_, notif_progress_gpu_));
        ASSERT_NO_FATAL_FAILURE(allocMapped(notif_send_, notif_send_gpu_));
    }

    void
    TearDown() override {
        if (running_) {
            const uint32_t control = DOCA_GPU_EXIT;
            EXPECT_EQ(cudaMemcpyAsync(&gpu_state_->control,
                                      &control,
                                      sizeof(control),
                                      cudaMemcpyHostToDevice,
                                      copy_stream_),
                      cudaSuccess);
            EXPECT_EQ(cudaStreamSynchronize(copy_stream_), cudaSuccess);
            EXPECT_EQ(cudaStreamSynchronize(progress_stream_), cudaSuccess);
        }
        for (Peer &peer : peers_) {
            cudaFree(peer.qp);
            cudaFree(peer.cqes);
            cudaFree(peer.dbrec);
        }
        for (void *host : {static_cast<void *>(host_state_),
                           static_cast<void *>(notif_fill_),
                           static_cast<void *>(notif_progress_),
                           static_cast<void *>(notif_send_)}) {
            if (host != nullptr) {
                cudaFreeHost(host);
            }
        }
        if (gpu_state_ != nullptr) {
            cudaFree(gpu_state_);
        }
        if (ring_ != nullptr) {
            cudaFree(ring_);
        }
        if (copy_stream_ != nullptr) {
            cudaStreamDestroy(copy_stream_);
        }
        if (progress_stream_ != nullptr) {
            cudaStreamDestroy(progress_stream_);
        }
    }

    template<typename T>
    void
    allocMapped(T *&host, T *&device) {
        ASSERT_EQ(cudaHostAlloc(reinterpret_cast<void **>(&host), sizeof(T), cudaHostAllocMapped),
                  cudaSuccess);
        std::memset(host, 0, sizeof(T));
        ASSERT_EQ(cudaHostGetDevicePointer(reinterpret_cast<void **>(&device), host, 0),
                  cudaSuccess);
    }

    // A notification QP with its own RQ CQ, receive queue already filled as notif_fill leaves it.
    void
    addPeer() {
        Peer &peer = peers_.emplace_back();
        ASSERT_EQ(cudaMalloc(&peer.qp, sizeof(doca_gpu_dev_verbs_qp)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&peer.cqes, rq_cqe_num * DOCA_GPUNETIO_VERBS_CQE_SIZE), cudaSuccess);
        // Owner bit set on every CQE: nothing completed in the first lap.
        ASSERT_EQ(cudaMemset(peer.cqes, 0xff, rq_cqe_num * DOCA_GPUNETIO_VERBS_CQE_SIZE),
                  cudaSuccess);
        ASSERT_EQ(cudaMalloc(&peer.dbrec, sizeof(*peer.dbrec)), cudaSuccess);

        doca_gpu_dev_verbs_qp qp{};
        qp.nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB;
        qp.rq_wqe_num = rq_cqe_num;
        qp.rq_wqe_mask = rq_cqe_num - 1;
        qp.rq_wqe_pi = rq_cqe_num;
        qp.rq_dbrec = peer.dbrec;
        qp.cq_rq.cqe_daddr = peer.cqes;
        qp.cq_rq.cqe_num = rq_cqe_num;
        qp.cq_rq.cqe_mask = rq_cqe_num - 1;
        qp.cq_rq.cqe_size = DOCA_GPUNETIO_VERBS_CQE_SIZE;
        qp.cq_rq.mem_type = DOCA_GPUNETIO_VERBS_MEM_TYPE_GPU;
        ASSERT_EQ(cudaMemcpy(peer.qp, &qp, sizeof(qp), cudaMemcpyHostToDevice), cudaSuccess);
    }

    void
    startProgress() {
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        ASSERT_EQ(doca_kernel_progress(progress_stream_,
                                       ring_,
                                       gpu_state_,
                                       notif_fill_gpu_,
                                       notif_progress_gpu_,
                                       notif_send_gpu_,
                                       host_state_gpu_),
                  DOCA_SUCCESS);
        running_ = true;
    }

    // Complete the peer's next receive as the NIC would, with the owner bit of the current lap.
    void
    deliver(Peer &peer) {
        const uint64_t index = peer.delivered++;
        doca_gpu_dev_verbs_cqe64 cqe{};
        cqe.wqe_counter = htobe16(static_cast<uint16_t>(index));
        cqe.op_own = (MLX5_CQE_RESP_SEND << DOCA_GPUNETIO_VERBS_MLX5_CQE_OPCODE_SHIFT) |
            ((index & rq_cqe_num) != 0 ? MLX5_CQE_OWNER_MASK : 0);
        uint8_t *slot = peer.cqes + (index & (rq_cqe_num - 1)) * DOCA_GPUNETIO_VERBS_CQE_SIZE;
        ASSERT_EQ(cudaMemcpyAsync(slot, &cqe, sizeof(cqe), cudaMemcpyHostToDevice, copy_stream_),
                  cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(copy_stream_), cudaSuccess);
    }

    // One getNotifs() step for a peer: hand its QP to the GPU and collect at most one message.
    void
    poll(Peer &peer, bool &received) {
        auto *mailbox = reinterpret_cast<volatile docaNotif *>(notif_progress_);
        mailbox->qp_gpu = peer.qp;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        const auto deadline = std::chrono::steady_clock::now() + mailbox_timeout;
        while (mailbox->qp_gpu != nullptr) {
            if (std::chrono::steady_clock::now() > deadline) {
                FAIL() << "progress kernel did not answer the notification mailbox";
            }
        }
        received = mailbox->msg_num != 0;
        if (received) {
            ++peer.received;
        }
        ASSERT_LE(peer.received, peer.delivered) << "notification reported before it arrived";
    }

    // Poll every peer until a whole pass reports nothing.
    void
    drain() {
        bool any = true;
        while (any) {
            any = false;
            for (Peer &peer : peers_) {
                bool received = false;
                ASSERT_NO_FATAL_FAILURE(poll(peer, received));
                any = any || received;
            }
        }
    }

    uint64_t
    rqProducerIndex(const Peer &peer) {
        uint64_t value = 0;
        EXPECT_EQ(
            cudaMemcpyAsync(
                &value, &peer.qp->rq_wqe_pi, sizeof(value), cudaMemcpyDeviceToHost, copy_stream_),
            cudaSuccess);
        EXPECT_EQ(cudaStreamSynchronize(copy_stream_), cudaSuccess);
        return value;
    }

    std::vector<Peer> peers_;
    docaHostState *host_state_ = nullptr;

private:
    docaHostState *host_state_gpu_ = nullptr;
    cudaStream_t progress_stream_ = nullptr;
    cudaStream_t copy_stream_ = nullptr;
    docaSlotDesc *ring_ = nullptr;
    docaGpuState *gpu_state_ = nullptr;
    docaNotif *notif_fill_ = nullptr;
    docaNotif *notif_fill_gpu_ = nullptr;
    docaNotif *notif_progress_ = nullptr;
    docaNotif *notif_progress_gpu_ = nullptr;
    docaNotif *notif_send_ = nullptr;
    docaNotif *notif_send_gpu_ = nullptr;
    bool running_ = false;
};

} // namespace

TEST_F(GpunetioNotifProgress, PeersKeepIndependentReceiveIndexes) {
    for (int i = 0; i < 3; ++i) {
        ASSERT_NO_FATAL_FAILURE(addPeer());
    }
    ASSERT_NO_FATAL_FAILURE(startProgress());

    // A message from one peer must not move another peer's receive index.
    ASSERT_NO_FATAL_FAILURE(deliver(peers_[0]));
    ASSERT_NO_FATAL_FAILURE(drain());
    ASSERT_NO_FATAL_FAILURE(deliver(peers_[1]));
    ASSERT_NO_FATAL_FAILURE(deliver(peers_[1]));
    ASSERT_NO_FATAL_FAILURE(deliver(peers_[2]));
    ASSERT_NO_FATAL_FAILURE(drain());

    // Interleave arrivals with one getNotifs() pass each, as a target serving several peers does.
    const int arrivals[] = {2, 0, 1, 1, 2, 0, 0, 2, 1, 2, 2, 0, 1, 0, 2, 1};
    for (int arrival : arrivals) {
        ASSERT_NO_FATAL_FAILURE(deliver(peers_[arrival]));
        for (Peer &peer : peers_) {
            bool received = false;
            ASSERT_NO_FATAL_FAILURE(poll(peer, received));
        }
    }
    ASSERT_NO_FATAL_FAILURE(drain());

    for (size_t i = 0; i < peers_.size(); ++i) {
        EXPECT_EQ(peers_[i].received, peers_[i].delivered) << "peer " << i;
        // Each consumed message re-posts one receive on that peer's own RQ.
        EXPECT_EQ(rqProducerIndex(peers_[i]), rq_cqe_num + peers_[i].received) << "peer " << i;
    }
    EXPECT_EQ(std::atomic_ref<uint32_t>(host_state_->failed).load(std::memory_order_acquire), 0U);
}
