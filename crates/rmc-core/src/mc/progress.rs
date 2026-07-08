use indicatif::{ProgressBar, ProgressStyle};

use super::run::SimulationCtx;
use super::traits::RunCallbacks;

#[derive(Clone, Debug)]
pub struct IndicatifProgress {
    bar: ProgressBar,
    last_steps_done: u64,
    finish_message: Option<String>,
}

impl IndicatifProgress {
    pub fn new(bar: ProgressBar) -> Self {
        Self {
            bar,
            last_steps_done: 0,
            finish_message: None,
        }
    }

    pub fn with_total(total_steps: u64, label: impl Into<String>) -> Self {
        let bar = ProgressBar::new(total_steps);
        bar.set_style(default_progress_style());
        bar.set_prefix(label.into());
        Self::new(bar)
    }

    pub fn with_finish_message(mut self, message: impl Into<String>) -> Self {
        self.finish_message = Some(message.into());
        self
    }

    pub fn bar(&self) -> &ProgressBar {
        &self.bar
    }

    pub fn finish(&self) {
        match &self.finish_message {
            Some(message) => self.bar.finish_with_message(message.clone()),
            None => self.bar.finish(),
        }
    }
}

impl RunCallbacks<SimulationCtx> for IndicatifProgress {
    fn on_cycle(&mut self, ctx: &SimulationCtx) {
        let delta = ctx.steps_done.saturating_sub(self.last_steps_done);
        if delta > 0 {
            self.bar.inc(delta);
            self.last_steps_done = ctx.steps_done;
        }
        self.bar.set_message(format!("cycle {}", ctx.cycles_done));
    }
}

impl Drop for IndicatifProgress {
    fn drop(&mut self) {
        self.finish();
    }
}

pub fn default_progress_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "{prefix:>10} [{elapsed_precise}] {wide_bar:.cyan/blue} {pos}/{len} steps ({eta}) {msg}",
    )
    .expect("progress template must be valid")
    .progress_chars("=>-")
}
