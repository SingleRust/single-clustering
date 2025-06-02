pub(crate) mod standard;
pub(crate) mod fast;

pub(crate) mod merging;

#[derive(Debug, Clone, Copy)]
pub enum QualityFunction {
    CPM,
    RBConfiguration,
}