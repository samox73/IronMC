use rmc_core::mc::{
    run_chain, Measurement, MetropolisKernel, NoopCallbacks, Runner, SimulationParams,
    SingleUpdateSet, Update,
};
use rmc_core::random::{uniform_index, ChainId, SeedSource};
use rmc_core::Merge;

#[derive(Clone, Debug)]
struct Ising2x2 {
    spins: [i8; 4],
}

impl Ising2x2 {
    fn ordered() -> Self {
        Self { spins: [1; 4] }
    }

    fn magnetization_per_spin(&self) -> f64 {
        self.spins.iter().map(|spin| f64::from(*spin)).sum::<f64>() / self.spins.len() as f64
    }

    fn flip(&mut self, idx: usize) {
        self.spins[idx] *= -1;
    }
}

#[derive(Clone, Debug)]
struct LazyInfiniteTemperatureFlip {
    proposed_idx: usize,
}

impl LazyInfiniteTemperatureFlip {
    fn new() -> Self {
        Self { proposed_idx: 0 }
    }
}

impl Update<Ising2x2> for LazyInfiniteTemperatureFlip {
    fn attempt<R: rand::Rng + ?Sized>(&mut self, state: &mut Ising2x2, rng: &mut R) -> f64 {
        self.proposed_idx = uniform_index(rng, state.spins.len());
        0.5
    }

    fn accept(&mut self, state: &mut Ising2x2) {
        state.flip(self.proposed_idx);
    }
}

#[derive(Clone, Debug, Default)]
struct MagnetizationMoments {
    samples: u64,
    signed_sum: f64,
    abs_sum: f64,
}

impl Measurement<Ising2x2> for MagnetizationMoments {
    type Output = Self;

    fn measure(&mut self, state: &Ising2x2) {
        let magnetization = state.magnetization_per_spin();
        self.samples += 1;
        self.signed_sum += magnetization;
        self.abs_sum += magnetization.abs();
    }

    fn finish(self) -> Self::Output {
        self
    }
}

impl MagnetizationMoments {
    fn mean_signed(&self) -> f64 {
        self.signed_sum / self.samples as f64
    }

    fn mean_abs(&self) -> f64 {
        self.abs_sum / self.samples as f64
    }
}

#[test]
fn infinite_temperature_ising_2x2_matches_exact_magnetization() {
    let mut rng = SeedSource::new(0x15_1eaf).rng_for(ChainId(0));
    let mut kernel =
        MetropolisKernel::new(SingleUpdateSet::new(LazyInfiniteTemperatureFlip::new()));

    let (_state, stats, moments) = run_chain(
        Ising2x2::ordered(),
        &mut rng,
        &mut kernel,
        MagnetizationMoments::default(),
        SimulationParams {
            max_steps: 200_000,
            steps_per_cycle: 4,
            cycles_per_check: 1,
        },
        NoopCallbacks,
    )
    .unwrap();

    assert_eq!(stats.cycles_done, 50_000);
    assert!(
        moments.mean_signed().abs() < 0.02,
        "signed magnetization should vanish at beta=0, got {}",
        moments.mean_signed()
    );
    assert!(
        (moments.mean_abs() - exact_infinite_temperature_abs_magnetization()).abs() < 0.02,
        "absolute magnetization mean={} expected={}",
        moments.mean_abs(),
        exact_infinite_temperature_abs_magnetization()
    );
}

fn exact_infinite_temperature_abs_magnetization() -> f64 {
    let mut total = 0.0;
    for mask in 0_u8..16 {
        let magnetization = (0..4)
            .map(|bit| if mask & (1 << bit) == 0 { -1.0 } else { 1.0 })
            .sum::<f64>()
            / 4.0;
        total += magnetization.abs();
    }
    total / 16.0
}

#[derive(Clone, Debug)]
struct IsingLattice {
    width: usize,
    height: usize,
    spins: Vec<i8>,
}

impl IsingLattice {
    fn ordered(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            spins: vec![1; width * height],
        }
    }

    fn len(&self) -> usize {
        self.spins.len()
    }

    fn magnetization_per_spin(&self) -> f64 {
        self.spins.iter().map(|spin| f64::from(*spin)).sum::<f64>() / self.len() as f64
    }

    fn delta_energy_for_flip(&self, idx: usize) -> i32 {
        let x = idx % self.width;
        let y = idx / self.width;
        let left = self.spin_at(wrap_sub(x, self.width), y);
        let right = self.spin_at((x + 1) % self.width, y);
        let up = self.spin_at(x, wrap_sub(y, self.height));
        let down = self.spin_at(x, (y + 1) % self.height);
        2 * i32::from(self.spins[idx]) * i32::from(left + right + up + down)
    }

    fn flip(&mut self, idx: usize) {
        self.spins[idx] *= -1;
    }

    fn spin_at(&self, x: usize, y: usize) -> i8 {
        self.spins[y * self.width + x]
    }
}

fn wrap_sub(value: usize, modulus: usize) -> usize {
    if value == 0 {
        modulus - 1
    } else {
        value - 1
    }
}

#[derive(Clone, Debug)]
struct MetropolisSpinFlip {
    beta: f64,
    proposed_idx: usize,
}

impl MetropolisSpinFlip {
    fn new(beta: f64) -> Self {
        Self {
            beta,
            proposed_idx: 0,
        }
    }
}

impl Update<IsingLattice> for MetropolisSpinFlip {
    fn attempt<R: rand::Rng + ?Sized>(&mut self, state: &mut IsingLattice, rng: &mut R) -> f64 {
        self.proposed_idx = uniform_index(rng, state.len());
        let delta_energy = state.delta_energy_for_flip(self.proposed_idx);
        if delta_energy <= 0 {
            1.0
        } else {
            (-self.beta * f64::from(delta_energy)).exp()
        }
    }

    fn accept(&mut self, state: &mut IsingLattice) {
        state.flip(self.proposed_idx);
    }
}

#[derive(Clone, Debug)]
struct BinderMeasurement {
    burn_in_cycles: u64,
    cycles_seen: u64,
    samples: u64,
    abs_m_sum: f64,
    m2_sum: f64,
    m4_sum: f64,
}

impl BinderMeasurement {
    fn new(burn_in_cycles: u64) -> Self {
        Self {
            burn_in_cycles,
            cycles_seen: 0,
            samples: 0,
            abs_m_sum: 0.0,
            m2_sum: 0.0,
            m4_sum: 0.0,
        }
    }
}

impl Measurement<IsingLattice> for BinderMeasurement {
    type Output = BinderSummary;

    fn measure(&mut self, state: &IsingLattice) {
        self.cycles_seen += 1;
        if self.cycles_seen <= self.burn_in_cycles {
            return;
        }

        let m = state.magnetization_per_spin();
        let m2 = m * m;
        self.samples += 1;
        self.abs_m_sum += m.abs();
        self.m2_sum += m2;
        self.m4_sum += m2 * m2;
    }

    fn finish(self) -> Self::Output {
        BinderSummary {
            samples: self.samples,
            abs_m_sum: self.abs_m_sum,
            m2_sum: self.m2_sum,
            m4_sum: self.m4_sum,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct BinderSummary {
    samples: u64,
    abs_m_sum: f64,
    m2_sum: f64,
    m4_sum: f64,
}

impl BinderSummary {
    fn mean_abs_m(&self) -> f64 {
        self.abs_m_sum / self.samples as f64
    }

    fn binder_cumulant(&self) -> f64 {
        let mean_m2 = self.m2_sum / self.samples as f64;
        let mean_m4 = self.m4_sum / self.samples as f64;
        1.0 - mean_m4 / (3.0 * mean_m2 * mean_m2)
    }
}

impl Merge for BinderSummary {
    fn merge(self, other: Self) -> Self {
        Self {
            samples: self.samples + other.samples,
            abs_m_sum: self.abs_m_sum + other.abs_m_sum,
            m2_sum: self.m2_sum + other.m2_sum,
            m4_sum: self.m4_sum + other.m4_sum,
        }
    }
}

fn run_ising_lattice(beta: f64) -> BinderSummary {
    const L: usize = 8;
    const SWEEPS: u64 = 2_500;
    const BURN_IN_SWEEPS: u64 = 500;

    let report = Runner::new(SeedSource::new(0x15_1eaf ^ beta.to_bits()), |_chain| {
        let state = IsingLattice::ordered(L, L);
        let update = MetropolisSpinFlip::new(beta);
        let kernel = MetropolisKernel::new(SingleUpdateSet::new(update));
        let measurement = BinderMeasurement::new(BURN_IN_SWEEPS);
        (state, kernel, measurement)
    })
    .chains(8)
    .run(SimulationParams {
        max_steps: SWEEPS * (L * L) as u64,
        steps_per_cycle: (L * L) as u64,
        cycles_per_check: 1,
    })
    .unwrap();

    report.output
}

#[test]
fn finite_size_ising_critical_region_has_expected_binder_signal() {
    let high_temperature = run_ising_lattice(0.20);
    let critical = run_ising_lattice(0.5 * (1.0 + 2.0_f64.sqrt()).ln());
    let low_temperature = run_ising_lattice(0.70);

    let high_binder = high_temperature.binder_cumulant();
    let critical_binder = critical.binder_cumulant();
    let low_binder = low_temperature.binder_cumulant();

    assert!(
        high_temperature.mean_abs_m() < critical.mean_abs_m(),
        "critical |m| should exceed high-temperature |m|: high={} critical={}",
        high_temperature.mean_abs_m(),
        critical.mean_abs_m()
    );
    assert!(
        critical.mean_abs_m() < low_temperature.mean_abs_m(),
        "low-temperature |m| should exceed critical |m|: critical={} low={}",
        critical.mean_abs_m(),
        low_temperature.mean_abs_m()
    );
    assert!(
        high_binder < critical_binder && critical_binder < low_binder,
        "Binder cumulant should rise through the critical region: high={high_binder} critical={critical_binder} low={low_binder}"
    );
    assert!(
        (0.50..0.68).contains(&critical_binder),
        "8x8 Ising Binder cumulant at beta_c should be near the known square-lattice crossing, got {critical_binder}"
    );
}
