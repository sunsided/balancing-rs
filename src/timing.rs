use std::fmt::{Debug, Display, Formatter};
use std::ops::{Add, AddAssign, Deref, Div, Mul};

#[derive(Copy, Clone, PartialOrd, PartialEq, Default)]
pub struct Seconds(pub f64);

#[derive(Copy, Clone, PartialOrd, PartialEq, Default)]
pub struct Milliseconds(pub f64);

#[derive(Copy, Clone, PartialOrd, PartialEq, Default)]
pub struct Microseconds(pub f64);

#[derive(Copy, Clone, PartialOrd, PartialEq, Default)]
pub struct Nanoseconds(pub f64);

impl Deref for Seconds {
    type Target = f64;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Deref for Milliseconds {
    type Target = f64;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Deref for Microseconds {
    type Target = f64;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Deref for Nanoseconds {
    type Target = f64;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Into<Seconds> for Milliseconds {
    fn into(self) -> Seconds {
        Seconds(self.0 * 1e-3)
    }
}

impl From<Seconds> for Milliseconds {
    fn from(seconds: Seconds) -> Self {
        Self(seconds.0 * 1e3)
    }
}

impl Into<Seconds> for Microseconds {
    fn into(self) -> Seconds {
        Seconds(self.0 * 1e-6)
    }
}

impl From<Seconds> for Microseconds {
    fn from(seconds: Seconds) -> Self {
        Self(seconds.0 * 1e6)
    }
}

impl Into<Seconds> for Nanoseconds {
    fn into(self) -> Seconds {
        Seconds(self.0 * 1e-9)
    }
}

impl From<Seconds> for Nanoseconds {
    fn from(seconds: Seconds) -> Self {
        Self(seconds.0 * 1e9)
    }
}

impl Add for Seconds {
    type Output = Seconds;

    fn add(self, rhs: Self) -> Self::Output {
        Seconds(self.0 + rhs.0)
    }
}

impl AddAssign for Seconds {
    fn add_assign(&mut self, rhs: Self) {
        *self = Seconds(self.0 + rhs.0);
    }
}

impl Mul<usize> for Seconds {
    type Output = Seconds;

    fn mul(self, rhs: usize) -> Self::Output {
        Seconds(self.0 * rhs as f64)
    }
}

impl Div<usize> for Seconds {
    type Output = Seconds;

    fn div(self, rhs: usize) -> Self::Output {
        Seconds(self.0 / rhs as f64)
    }
}

impl Debug for Seconds {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} s", self.0)
    }
}

impl Debug for Milliseconds {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ms", self.0)
    }
}

impl Debug for Microseconds {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} µs", self.0)
    }
}

impl Debug for Nanoseconds {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ns", self.0)
    }
}

impl Display for Seconds {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} s", self.0)
    }
}

impl Display for Milliseconds {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ms", self.0)
    }
}

impl Display for Microseconds {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} µs", self.0)
    }
}

impl Display for Nanoseconds {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ns", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ns_in_sec() {
        let ns = Nanoseconds(350.877193);
        let s: Seconds = ns.into();
        assert_eq!(*s, 3.50877193e-7);
    }
}
