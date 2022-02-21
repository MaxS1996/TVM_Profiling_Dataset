# TVM_Profiling_Dataset
The datasets that have been generated by profiling typical TVM layer workloads on different target devices

| Target Device         | Compiler Backend | Conv2d            | Dense         | AvgPool2d    | MaxPool2d  | Dilated Conv2d | Depthwise Conv2d |
| --------------------- | ---------------- | :---------------: | :-----------: | :----------: | :--------: | :------------: | :--------------: |
| Tesla A100            | TVM 0.9 - cuda   | **3218/3218**     | 9415/5014     | 2057/?       | 2068/?     | 5880/2710      | 5050/2710        |
| Tesla K80             | TVM 0.9 - cuda   | **2707/3218**     | 10075/5014    | 1967/?       | 1945/?     | 5040/2710      | 5050/2710        |
| Intel Xeon E5-2680    | TVM 0.9 - llvm   | **2593/2710**     | 5220/5014     | 0956/?       | 0951/?     | 3171/2710      | 3304/2710        |
