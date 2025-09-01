# 1.软件安装

## 1.1 下载安装包

官网：https://alien-project.org/

github：https://github.com/chrxh/alien?tab=readme-ov-file

## 1.2 配置环境

先决条件：[CUDA Toolkit 11.2+](https://developer.nvidia.com/cuda-downloads)和 CMake 工具链

CUDA Toolkit 下载：https://developer.nvidia.com/cuda-toolkit-archive 选择11.2+版本 12.5以下

Cmake下载：https://cmake.org/download/

VS2022安装

以下是基于您提供的截图内容，在 Windows 系统下构建该源码的详细步骤：

------

### **1.2.1 安装先决条件**

在开始之前，请确保系统满足以下条件：

- **CUDA Toolkit 11.2+**（可从 [CUDA Toolkit 官网](https://developer.nvidia.com/cuda-toolkit) 下载并安装）。
- **CMake**（从 [CMake 官网](https://cmake.org/download/) 下载并安装）。
- **Visual Studio 2022 **（确保安装了 C++ 开发工具链）。

------

### **1.2.2 下载源码**

1. 打开命令提示符（`cmd`）或 PowerShell，切换到目标文件夹（用于存放源码的地方），例如：

   ```bash
   cd C:\Users\11618
   ```

2. 克隆仓库并递归下载子模块：

   ```bash
   git clone --recursive https://github.com/chrxh/alien.git
   ```

   如果已经克隆了仓库但子模块未更新，请运行：

   ```bash
   git pull --recurse-submodules
   ```

------

### **1.2.3 创建构建目录**

1. 进入下载的源码目录：

   ```bash
   cd alien
   ```

2. 创建并进入 `build` 文件夹：

   ```bash
   mkdir build && cd build
   ```

------

### **1.2.4 配置 CMake**

运行以下命令：

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release
```

这一步将生成用于 Visual Studio 的解决方案文件。

------

### **1.2.5 构建项目**

1. 使用 CMake 构建项目：

   ```bash
   cmake --build . --config Release -j8
   ```

   - `--config Release` 表示构建 Release 模式。
   - `-j8` 表示使用 8 个线程进行并行编译。

------

### **1.2.6 运行可执行文件**

如果一切顺利，可以在 `build` 文件夹下找到编译好的可执行文件，比如：

```bash
.\Release\alien.exe
```



