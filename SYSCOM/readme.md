# Systems and Communications Admin


## Your task is to create your own 32-bit Operating System. 

### Success Criterion 

- **Bootloader & Core Kernel**:  Write a bootloader in **x86 Assembly** to initialize the CPU and execute a minimal **C or C++ kernel**. The goal is to successfully boot into your own code within an emulator and display a startup message on the screen.

- **Interactive Kernel & Shell**: Implement basic drivers for screen and keyboard I/O by handling hardware interrupts. Use this functionality to build an interactive shell that accepts user input and executes essential commands like `echo`, `clear`, and `help`.

- **Filesystem Implementation**:  Design the data structures for a simple, read-only filesystem loaded from an **initial ramdisk (initrd)**. Extend the shell with the ls and cat commands to list the files on the ramdisk and display their contents.

- **Create a Project Showcase Video**: Produce a short **(<5 min)** video that demonstrates the finished OS, from booting to using all shell commands. The video should also cover potential future improvements upon the project. 
