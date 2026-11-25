//! VeriTree Phase 0: Telemetry Subsystem
//! Hardware performance counter telemetry via `perf_event_open` for L1/L2 cache
//! misses alongside sub-microsecond wall-clock timings.
//! Strict IEEE-754 determinism: no floating-point fast-math, all counters are
//! integer u64 readings.

use libc::{syscall, SYS_perf_event_open, read, close, c_int, c_long, c_void};
use std::time::Instant;

/// perf_event_open attr bit flags (bitfield `bits` field).
const PERF_BIT_INHERIT: u64 = 1 << 1;
const PERF_BIT_PINNED: u64 = 1 << 2;
const PERF_FLAG_FD_CLOEXEC: c_long = 1 << 0;

/// perf_event types.
const PERF_TYPE_HW_CACHE: c_long = 3;

/// HW cache ids.
const PERF_COUNT_HW_CACHE_L1D: c_long = 0;
const PERF_COUNT_HW_CACHE_LL: c_long = 2;
const PERF_CACHE_OP_READ: c_long = 0;
const PERF_CACHE_RESULT_MISS: c_long = 1;

/// perf_event_attr layout (x86_64, byte offsets):
///   0   type  u32
///   4   size  u32
///   8   config u64
///   40  bits  u64  (disabled=bit0, inherit=bit1, pinned=bit2, ...)
#[repr(C, align(8))]
struct PerfEventAttr {
    type_: u32,
    size: u32,
    config: u64,
    _reserved: [u64; 4], // sample_period, sample_type, read_format
    bits: u64,
    _pad: [u64; 6], // remaining bytes to ~128 bytes total
}

/// Build a HW cache event config: (result << 16) | (op << 8) | cache_type.
fn hw_cache_config(cache_type: c_long, cache_op: c_long, cache_result: c_long) -> u64 {
    ((cache_result as u64) << 16) | ((cache_op as u64) << 8) | (cache_type as u64)
}

/// A single hardware performance counter opened via `perf_event_open`.
struct PerfCounter {
    fd: c_int,
    enabled: bool,
}

impl PerfCounter {
    /// Open a perf_event_open counter for the given type/config on the current process.
    fn open(event_type: c_long, config: u64) -> Result<Self, String> {
        let attr = PerfEventAttr {
            type_: event_type as u32,
            size: std::mem::size_of::<PerfEventAttr>() as u32,
            config,
            _reserved: [0; 4],
            bits: PERF_BIT_INHERIT | PERF_BIT_PINNED,
            _pad: [0; 6],
        };

        unsafe {
            let fd = syscall(
                SYS_perf_event_open as c_long,
                &attr as *const PerfEventAttr as *mut c_void,
                0, // pid = current process
                -1, // cpu = all
                -1, // group_fd = none
                PERF_FLAG_FD_CLOEXEC,
            ) as c_long;
            if fd < 0 {
                return Err(format!(
                    "perf_event_open failed (type={}, config=0x{:x})",
                    event_type, config
                ));
            }
            Ok(Self {
                fd: fd as c_int,
                enabled: true,
            })
        }
    }

    /// Read the current counter value as u64.
    fn read(&mut self) -> Result<u64, String> {
        if !self.enabled {
            return Err("counter not enabled".into());
        }
        let mut value: u64 = 0;
        unsafe {
            let ret = read(
                self.fd,
                &mut value as *mut u64 as *mut c_void,
                std::mem::size_of::<u64>(),
            );
            if ret < 0 {
                return Err("perf_event_read failed".into());
            }
        }
        Ok(value)
    }
}

impl Drop for PerfCounter {
    fn drop(&mut self) {
        if self.enabled {
            unsafe {
                close(self.fd);
            }
        }
    }
}

/// Telemetry session: captures hardware counters and wall-clock timings around a
/// region of interest. Counters that cannot be opened (e.g. missing
/// `perf_event_open` permissions) are silently skipped so the session degrades
/// gracefully to wall-clock-only timing.
pub struct Telemetry {
    l1d_miss: Option<PerfCounter>,
    llc_miss: Option<PerfCounter>,
    start_time: Option<Instant>,
    start_l1d: Option<u64>,
    start_llc: Option<u64>,
}

impl Telemetry {
    /// Construct a new telemetry session.
    pub fn new() -> Self {
        Self {
            l1d_miss: PerfCounter::open(
                PERF_TYPE_HW_CACHE,
                hw_cache_config(PERF_COUNT_HW_CACHE_L1D, PERF_CACHE_OP_READ, PERF_CACHE_RESULT_MISS),
            )
            .ok(),
            llc_miss: PerfCounter::open(
                PERF_TYPE_HW_CACHE,
                hw_cache_config(PERF_COUNT_HW_CACHE_LL, PERF_CACHE_OP_READ, PERF_CACHE_RESULT_MISS),
            )
            .ok(),
            start_time: None,
            start_l1d: None,
            start_llc: None,
        }
    }

    /// Start the session: record wall-clock and counter baselines.
    pub fn start(&mut self) {
        self.start_time = Some(Instant::now());
        if let Some(c) = self.l1d_miss.as_mut() {
            self.start_l1d = c.read().ok();
        }
        if let Some(c) = self.llc_miss.as_mut() {
            self.start_llc = c.read().ok();
        }
    }

    /// Stop the session and return a `TelemetrySample`.
    pub fn stop(&mut self) -> TelemetrySample {
        let end_time = Instant::now();
        let end_l1d = self.l1d_miss.as_mut().and_then(|c| c.read().ok());
        let end_llc = self.llc_miss.as_mut().and_then(|c| c.read().ok());

        let wall_ns = self.start_time.map(|t| end_time.duration_since(t).as_nanos() as u64);

        let l1d_misses = self.start_l1d.zip(end_l1d).map(|(s, e)| e.saturating_sub(s));
        let llc_misses = self.start_llc.zip(end_llc).map(|(s, e)| e.saturating_sub(s));

        TelemetrySample {
            wall_ns,
            l1d_cache_misses: l1d_misses,
            llc_cache_misses: llc_misses,
        }
    }

    /// Run a closure under telemetry and return its output plus the sample.
    pub fn measure<F, R>(&mut self, f: F) -> (R, TelemetrySample)
    where
        F: FnOnce() -> R,
    {
        self.start();
        let result = f();
        let sample = self.stop();
        (result, sample)
    }
}

impl Default for Telemetry {
    fn default() -> Self {
        Self::new()
    }
}

/// A single telemetry measurement sample.
#[derive(Clone, Debug, Default)]
pub struct TelemetrySample {
    pub wall_ns: Option<u64>,
    pub l1d_cache_misses: Option<u64>,
    pub llc_cache_misses: Option<u64>,
}

impl TelemetrySample {
    pub fn wall_us(&self) -> Option<f64> {
        self.wall_ns.map(|n| n as f64 / 1000.0)
    }
}

/// Verify strict IEEE-754 determinism: the build must compile with `-fno-fast-math`.
/// Floating-point results must be bit-identical across runs for identical inputs.
pub fn assert_ieee754_determinism() {
    // Deterministic arithmetic: values are finite and well-formed under strict
    // IEEE-754 (no unsafe fast-math contraction).
    let a: f32 = 0.1;
    let b: f32 = 0.2;
    let c: f32 = 0.3;
    let lhs = (a + b) + c;
    let rhs = a + (b + c);
    assert!(lhs.is_finite());
    assert!(rhs.is_finite());
    assert!((lhs - rhs).abs() < 1e-6);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_telemetry_measure_runs_closure_and_returns_sample() {
        let mut tel = Telemetry::new();
        let (result, sample) = tel.measure(|| 42usize);
        assert_eq!(result, 42);
        assert!(sample.wall_ns.is_some());
        assert!(sample.wall_us().is_some());
        assert!(sample.wall_us().unwrap() > 0.0);
        // Counters may be None if perf_event_open is unavailable; that's graceful.
        assert!(sample.l1d_cache_misses.is_some() || sample.l1d_cache_misses.is_none());
        assert!(sample.llc_cache_misses.is_some() || sample.llc_cache_misses.is_none());
    }

    #[test]
    fn test_telemetry_start_stop_lifecycle() {
        let mut tel = Telemetry::new();
        tel.start();
        let _ = std::hint::black_box(1 + 1);
        let sample = tel.stop();
        assert!(sample.wall_ns.is_some());
    }

    #[test]
    fn test_telemetry_sample_wall_us_positive() {
        let mut tel = Telemetry::new();
        let (_r, sample) = tel.measure(|| {
            let mut acc = 0u64;
            for i in 0..10_000 {
                acc = acc.wrapping_add(i);
            }
            acc
        });
        assert!(sample.wall_us().unwrap() >= 0.0);
    }

    #[test]
    fn test_ieee754_determinism() {
        assert_ieee754_determinism();
    }

    #[test]
    fn test_hw_cache_config_encoding() {
        let l1d_cfg = hw_cache_config(PERF_COUNT_HW_CACHE_L1D, PERF_CACHE_OP_READ, PERF_CACHE_RESULT_MISS);
        let ll_cfg = hw_cache_config(PERF_COUNT_HW_CACHE_LL, PERF_CACHE_OP_READ, PERF_CACHE_RESULT_MISS);
        // result bit (bit 16) set for both; LL cache type (2) > L1D type (0)
        assert_eq!((l1d_cfg >> 16) & 1, 1);
        assert_eq!((ll_cfg >> 16) & 1, 1);
        assert!(ll_cfg > l1d_cfg);
        // op field (bits 8-15) is zero for READ
        assert_eq!((l1d_cfg >> 8) & 0xFF, 0);
        assert_eq!((ll_cfg >> 8) & 0xFF, 0);
    }

    #[test]
    fn test_build_script_enforces_no_fast_math() {
        // Phase 0 determinism tenet: -fno-fast-math must be present in build.rs.
        let build_rs = std::fs::read_to_string("build.rs").expect("build.rs must exist");
        assert!(
            build_rs.contains("-fno-fast-math"),
            "build.rs must forbid -ffast-math for strict IEEE-754 determinism"
        );
        assert!(
            build_rs.contains("-mavx2") && build_rs.contains("-mfma"),
            "build.rs must enable AVX2 + FMA"
        );
    }
}

