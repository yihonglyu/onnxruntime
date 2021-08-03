// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <benchmark/benchmark.h>

#include <Windows.h>

int main(int argc, char** argv) {
  int affinity = -1;
  if (argc > 1) {
    // hack: if first parameter is an integer, it is the core
    // that we should be running on.
    try {
      int coreid = std::stoi(argv[1]);
      if (coreid >= 0 && coreid <= 64) {
        argv++;
        argc--;
        HANDLE process = GetCurrentProcess();
        DWORD_PTR processAffinityMask = (DWORD_PTR)1 << coreid;

        BOOL success = SetProcessAffinityMask(process, processAffinityMask);

        if (!success) {
          printf("Failed to set process affinity to core id: %d\n", coreid);        
        }
      }
    } catch (std::exception const& e) {
        // no parameter specified as core id
    }
  }

  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
}