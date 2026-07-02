use crate::Result;

/// Object-safe named-result sink.
///
/// Values are serialized through `erased_serde`, which keeps sink measurements output-only without
/// requiring `Any` or downcast.
pub trait ResultSink {
    /// Write `value` under `path`.
    ///
    /// Sink implementations should reject duplicate paths.
    fn put(&mut self, path: &str, value: &dyn erased_serde::Serialize) -> Result<()>;
}

/// A measurement that emits named results into a [`ResultSink`].
///
/// This is generic over the concrete simulation state, but object-safe for
/// `dyn SinkMeasurement<State>`.
pub trait SinkMeasurement<State> {
    fn name(&self) -> &str;
    fn measure(&mut self, state: &State);
    fn write_result(&self, sink: &mut dyn ResultSink) -> Result<()>;
}

/// Sink view that prefixes all keys with the emitting measurement name.
pub struct ScopedResultSink<'a> {
    inner: &'a mut dyn ResultSink,
    prefix: &'a str,
}

impl<'a> ScopedResultSink<'a> {
    pub fn new(inner: &'a mut dyn ResultSink, prefix: &'a str) -> Self {
        Self { inner, prefix }
    }
}

impl ResultSink for ScopedResultSink<'_> {
    fn put(&mut self, key: &str, value: &dyn erased_serde::Serialize) -> Result<()> {
        let path = format!("{}/{}", self.prefix, key);
        self.inner.put(&path, value)
    }
}
