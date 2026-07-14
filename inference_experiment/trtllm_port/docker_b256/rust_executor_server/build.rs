fn main() {
    println!("cargo:rustc-link-search=native=/build/rust/native");
    println!("cargo:rustc-link-lib=dylib=indicxlit_trtllm_bridge");
    println!("cargo:rustc-link-arg=-Wl,--allow-shlib-undefined");
    println!("cargo:rerun-if-changed=native/libindicxlit_trtllm_bridge.so");
}
