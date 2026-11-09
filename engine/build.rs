//! VeriTree Engine build script: compiles C++20 AVX2 kernels via cc
//! Compiler & SIMD Flags: -O3 -march=native -mavx2 -mfma -std=c++20 -fno-fast-math

fn main() {
    println!("cargo:rerun-if-changed=csrc/ring_buffer.hpp");
    println!("cargo:rerun-if-changed=csrc/avx2_cert_eval.cpp");
    println!("cargo:rerun-if-changed=csrc/corouting_tensor.cpp");

    cc::Build::new()
        .cpp(true)
        .std("c++20")
        .opt_level(3)
        .flag("-march=native")
        .flag("-mavx2")
        .flag("-mfma")
        .flag("-fno-fast-math") // Strict IEEE-754 Determinism
        .file("csrc/avx2_cert_eval.cpp")
        .file("csrc/corouting_tensor.cpp")
        .include("csrc")
        .compile("veritree_csrc");
}

