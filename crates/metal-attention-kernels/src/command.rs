//! Command buffer management with triple buffering.
//!
//! Triple buffering via dispatch_semaphore(3) ensures CPU and GPU can work
//! concurrently on different frames without stalling. The CPU can prepare
//! up to 2 frames ahead while the GPU processes the current one.

use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer, MTLCommandQueue};

/// Command buffer manager with triple buffering.
///
/// Uses a counting semaphore with value 3 to allow up to 3 in-flight
/// command buffers. This prevents the CPU from racing too far ahead
/// while keeping the GPU busy.
pub struct CommandManager {
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    semaphore: dispatch::Semaphore,
    max_in_flight: u32,
}

// SAFETY: MTLCommandQueue is thread-safe and dispatch::Semaphore is thread-safe.
unsafe impl Send for CommandManager {}
unsafe impl Sync for CommandManager {}

impl CommandManager {
    /// Create a new command manager with triple buffering (3 in-flight frames).
    pub fn new(queue: Retained<ProtocolObject<dyn MTLCommandQueue>>) -> Self {
        Self::with_in_flight(queue, 3)
    }

    /// Create a command manager with a custom number of in-flight frames.
    pub fn with_in_flight(
        queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
        max_in_flight: u32,
    ) -> Self {
        let semaphore = dispatch::Semaphore::new(max_in_flight);
        Self {
            queue,
            semaphore,
            max_in_flight,
        }
    }

    /// Wait for a frame slot and create a new command buffer.
    ///
    /// Blocks if all `max_in_flight` frames are still in-flight on the GPU.
    /// Returns a fresh command buffer ready for encoding.
    pub fn begin_frame(&self) -> Retained<ProtocolObject<dyn MTLCommandBuffer>> {
        self.semaphore.wait();
        self.queue
            .commandBuffer()
            .expect("Failed to create command buffer")
    }

    /// Commit a command buffer and signal the semaphore on completion.
    ///
    /// Adds a completion handler that signals the semaphore when the GPU
    /// finishes this command buffer, freeing a frame slot for `begin_frame`.
    pub fn end_frame(&self, cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>) {
        let sem = self.semaphore.clone();
        let block =
            block2::RcBlock::new(move |_buf: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
                sem.signal();
            });
        unsafe {
            cmd_buf.addCompletedHandler(block2::RcBlock::as_ptr(&block));
        }
        cmd_buf.commit();
    }

    /// Maximum number of in-flight frames.
    pub fn max_in_flight(&self) -> u32 {
        self.max_in_flight
    }

    /// Get a reference to the underlying command queue.
    pub fn queue(&self) -> &ProtocolObject<dyn MTLCommandQueue> {
        &self.queue
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::GpuDevice;

    #[test]
    fn test_command_manager_basic() {
        let gpu = GpuDevice::new();
        let mgr = CommandManager::new(gpu.command_queue.clone());
        assert_eq!(mgr.max_in_flight(), 3);

        // Begin and end a single frame
        let cmd_buf = mgr.begin_frame();
        mgr.end_frame(&cmd_buf);

        // Wait for GPU to finish
        cmd_buf.waitUntilCompleted();
    }

    #[test]
    fn test_command_manager_multiple_frames() {
        let gpu = GpuDevice::new();
        let mgr = CommandManager::new(gpu.command_queue.clone());

        // Submit 3 frames (should not block since triple buffered)
        let mut bufs = Vec::new();
        for _ in 0..3 {
            let cmd_buf = mgr.begin_frame();
            mgr.end_frame(&cmd_buf);
            bufs.push(cmd_buf);
        }

        // Wait for all to complete
        for buf in &bufs {
            buf.waitUntilCompleted();
        }
    }
}
