/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gpunetio_backend_aux.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

TEST(GpunetioParams, CudaStreamsDefaultsToFullPool) {
    EXPECT_EQ(nixl::gpunetio::parseCudaStreams(""), DOCA_POST_STREAM_NUM);
}

TEST(GpunetioParams, CudaStreamsAcceptsEveryPoolSize) {
    for (uint32_t count = 1; count <= DOCA_POST_STREAM_NUM; ++count) {
        EXPECT_EQ(nixl::gpunetio::parseCudaStreams(std::to_string(count)), count);
    }
}

TEST(GpunetioParams, CudaStreamsRejectsValuesOutsidePool) {
    for (const char *value : {"0", "5", "8", "-1", "2x", "x", " 2", "4294967296"}) {
        EXPECT_THROW(nixl::gpunetio::parseCudaStreams(value), std::invalid_argument) << value;
    }
}

TEST(GpunetioParams, DescriptorMustFitOneWqe) {
    EXPECT_TRUE(nixl::gpunetio::fitsSingleWqe(0));
    EXPECT_TRUE(nixl::gpunetio::fitsSingleWqe(DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE));
    EXPECT_FALSE(nixl::gpunetio::fitsSingleWqe(DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE + 1));
    // 4 GiB + 4 KiB would wrap the 32-bit WQE byte count to 4 KiB.
    EXPECT_FALSE(nixl::gpunetio::fitsSingleWqe((size_t{4} << 30) + 4096));
}

TEST(GpunetioIbDevice, NamePrefixDoesNotOpenAnotherDevice) {
    int count = 0;
    ibv_device **devices = ibv_get_device_list(&count);
    std::vector<std::string> names;
    for (int i = 0; devices != nullptr && i < count; ++i) {
        names.emplace_back(ibv_get_device_name(devices[i]));
    }
    if (devices != nullptr) {
        ibv_free_device_list(devices);
    }
    if (names.empty()) {
        GTEST_SKIP() << "No RDMA device available";
    }

    // A proper prefix of a device name that is not a device itself, such as mlx5_ for mlx5_0.
    std::string prefix = names.front();
    do {
        prefix.pop_back();
    } while (!prefix.empty() && std::find(names.begin(), names.end(), prefix) != names.end());
    if (prefix.empty()) {
        GTEST_SKIP() << "No device name has a usable prefix";
    }

    doca_verbs_context *context = open_ib_device(prefix.data());
    if (context != nullptr) {
        doca_verbs_context_destroy(context);
    }
    EXPECT_EQ(context, nullptr) << "'" << prefix << "' opened a device with a longer name";
}
