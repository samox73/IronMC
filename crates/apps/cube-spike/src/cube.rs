use cubecl::features::AtomicUsage;
use cubecl::prelude::*;

#[cube(launch)]
fn exp_kernel(input: &Array<f64>, output: &mut Array<f64>, lambda: f64) {
    if ABSOLUTE_POS < input.len() {
        output[ABSOLUTE_POS] = (-1.0 * lambda * input[ABSOLUTE_POS]).exp();
    }
}

#[cube(launch)]
fn trait_kernel<M: RatioModel>(input: &Array<f64>, output: &mut Array<f64>) {
    if ABSOLUTE_POS < input.len() {
        output[ABSOLUTE_POS] = M::ratio(input[ABSOLUTE_POS]);
    }
}

#[cube]
trait RatioModel: 'static + Send + Sync {
    fn ratio(x: f64) -> f64;
}

struct ToyModel;

#[cube]
impl RatioModel for ToyModel {
    fn ratio(x: f64) -> f64 {
        (x + 1.0) / (x + 2.0)
    }
}

#[cube(launch)]
fn loop_branch_kernel(input: &Array<u32>, output: &mut Array<u32>, iterations: u32) {
    if ABSOLUTE_POS < input.len() {
        let mut acc = input[ABSOLUTE_POS];
        let mut i = 0;
        while i < iterations + (ABSOLUTE_POS as u32 % 3) {
            acc = if (i + ABSOLUTE_POS as u32) % 2 == 0 {
                acc + 3
            } else {
                acc + 1
            };
            i += 1;
        }
        output[ABSOLUTE_POS] = acc;
    }
}

#[cube(launch)]
fn atomic_f64_add_kernel(output: &mut Array<Atomic<f64>>) {
    if UNIT_POS == 0 {
        output[0].fetch_add(1.0);
    }
}

#[cube]
fn philox_mul_hi(a: u32, b: u32) -> u32 {
    ((u64::cast_from(a) * u64::cast_from(b)) >> 32) as u32
}

#[cube]
fn philox_mul_lo(a: u32, b: u32) -> u32 {
    (u64::cast_from(a) * u64::cast_from(b)) as u32
}

#[cube(launch)]
fn philox_kernel(
    output: &mut Array<u32>,
    mut c0: u32,
    mut c1: u32,
    mut c2: u32,
    mut c3: u32,
    mut k0: u32,
    mut k1: u32,
) {
    if UNIT_POS == 0 {
        #[unroll]
        for round in 0..10 {
            if comptime![round > 0] {
                k0 += 0x9e37_79b9u32;
                k1 += 0xbb67_ae85u32;
            }
            let hi0 = philox_mul_hi(0xd251_1f53u32, c0);
            let lo0 = philox_mul_lo(0xd251_1f53u32, c0);
            let hi1 = philox_mul_hi(0xcd9e_8d57u32, c2);
            let lo1 = philox_mul_lo(0xcd9e_8d57u32, c2);
            let n0 = hi1 ^ c1 ^ k0;
            let n1 = lo1;
            let n2 = hi0 ^ c3 ^ k1;
            let n3 = lo0;
            c0 = n0;
            c1 = n1;
            c2 = n2;
            c3 = n3;
        }
        output[0] = c0;
        output[1] = c1;
        output[2] = c2;
        output[3] = c3;
    }
}

#[cube(launch)]
fn philox_words_kernel(output: &mut Array<u32>, blocks: usize) {
    if ABSOLUTE_POS < blocks {
        let mut c0 = ABSOLUTE_POS as u32;
        let mut c1 = 0u32;
        let mut c2 = 0u32;
        let mut c3 = 0u32;
        let mut k0 = 0u32;
        let mut k1 = 0u32;
        #[unroll]
        for round in 0..10 {
            if comptime![round > 0] {
                k0 += 0x9e37_79b9u32;
                k1 += 0xbb67_ae85u32;
            }
            let hi0 = philox_mul_hi(0xd251_1f53u32, c0);
            let lo0 = philox_mul_lo(0xd251_1f53u32, c0);
            let hi1 = philox_mul_hi(0xcd9e_8d57u32, c2);
            let lo1 = philox_mul_lo(0xcd9e_8d57u32, c2);
            let n0 = hi1 ^ c1 ^ k0;
            let n1 = lo1;
            let n2 = hi0 ^ c3 ^ k1;
            let n3 = lo0;
            c0 = n0;
            c1 = n1;
            c2 = n2;
            c3 = n3;
        }
        let out = ABSOLUTE_POS * 4;
        output[out] = c0;
        output[out + 1] = c1;
        output[out + 2] = c2;
        output[out + 3] = c3;
    }
}

pub fn run_exp<R: Runtime>(client: ComputeClient<R>, lambda: f64, dtau: &[f64]) -> Vec<f64> {
    let input = client.create_from_slice(f64::as_bytes(dtau));
    let output = client.empty(dtau.len() * core::mem::size_of::<f64>());
    exp_kernel::launch::<R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(dtau.len() as u32),
        unsafe { ArrayArg::from_raw_parts(input, dtau.len()) },
        unsafe { ArrayArg::from_raw_parts(output.clone(), dtau.len()) },
        lambda,
    );
    f64::from_bytes(&client.read_one_unchecked(output)).to_vec()
}

pub fn run_trait<R: Runtime>(client: ComputeClient<R>, input: &[f64]) -> Vec<f64> {
    let input_handle = client.create_from_slice(f64::as_bytes(input));
    let output = client.empty(input.len() * core::mem::size_of::<f64>());
    trait_kernel::launch::<ToyModel, R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(input.len() as u32),
        unsafe { ArrayArg::from_raw_parts(input_handle, input.len()) },
        unsafe { ArrayArg::from_raw_parts(output.clone(), input.len()) },
    );
    f64::from_bytes(&client.read_one_unchecked(output)).to_vec()
}

pub fn run_loop_branch<R: Runtime>(
    client: ComputeClient<R>,
    input: &[u32],
    iterations: u32,
) -> Vec<u32> {
    let input_handle = client.create_from_slice(u32::as_bytes(input));
    let output = client.empty(input.len() * core::mem::size_of::<u32>());
    loop_branch_kernel::launch::<R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(input.len() as u32),
        unsafe { ArrayArg::from_raw_parts(input_handle, input.len()) },
        unsafe { ArrayArg::from_raw_parts(output.clone(), input.len()) },
        iterations,
    );
    u32::from_bytes(&client.read_one_unchecked(output)).to_vec()
}

pub fn supports_atomic_f64_add<R: Runtime>(client: &ComputeClient<R>) -> bool {
    let ty = StorageType::Atomic(f64::as_type_native_unchecked().elem_type());
    client
        .properties()
        .atomic_type_usage(Type::new(ty))
        .contains(AtomicUsage::Add)
}

pub fn run_atomic_f64_add<R: Runtime>(client: ComputeClient<R>) -> Option<f64> {
    if !supports_atomic_f64_add(&client) {
        return None;
    }
    let output = client.create_from_slice(f64::as_bytes(&[0.0]));
    atomic_f64_add_kernel::launch::<R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        unsafe { ArrayArg::from_raw_parts(output.clone(), 1) },
    );
    Some(f64::from_bytes(&client.read_one_unchecked(output))[0])
}

pub fn run_philox<R: Runtime>(client: ComputeClient<R>, ctr: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let output = client.empty(4 * core::mem::size_of::<u32>());
    philox_kernel::launch::<R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        unsafe { ArrayArg::from_raw_parts(output.clone(), 4) },
        ctr[0],
        ctr[1],
        ctr[2],
        ctr[3],
        key[0],
        key[1],
    );
    let bytes = client.read_one_unchecked(output);
    let out = u32::from_bytes(&bytes);
    [out[0], out[1], out[2], out[3]]
}

pub fn run_philox_words<R: Runtime>(client: ComputeClient<R>, words: usize) -> Vec<u32> {
    let blocks = words.div_ceil(4);
    let output = client.empty(blocks * 4 * core::mem::size_of::<u32>());
    philox_words_kernel::launch::<R>(
        &client,
        CubeCount::Static(blocks as u32, 1, 1),
        CubeDim::new_1d(1),
        unsafe { ArrayArg::from_raw_parts(output.clone(), blocks * 4) },
        blocks,
    );
    let bytes = client.read_one_unchecked(output);
    u32::from_bytes(&bytes)[..words].to_vec()
}
