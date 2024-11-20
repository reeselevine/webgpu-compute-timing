(function () {
  try {
    let gpuDevice = null;

    // Hook into requestDevice to capture GPUDevice and enable timestamp-query
    const originalRequestDevice = GPUAdapter.prototype.requestDevice;
    GPUAdapter.prototype.requestDevice = async function (descriptor = {}) {
      const requiredFeatures = ["timestamp-query"];
      descriptor.requiredFeatures = [
        ...(descriptor.requiredFeatures || []),
        ...requiredFeatures,
      ];

      gpuDevice = await originalRequestDevice.apply(this, [descriptor]);
      console.log("Captured GPUDevice with required features:", gpuDevice.features);
      return gpuDevice;
    };

    const pipelineEntrypoints = new WeakMap(); // Store pipeline -> entrypoint mapping

    const originalCreateComputePipeline = GPUDevice.prototype.createComputePipeline;

    GPUDevice.prototype.createComputePipeline = function (descriptor) {
      const pipeline = originalCreateComputePipeline.call(this, descriptor);
      const entryPoint = descriptor?.compute?.entryPoint || "unknown";
      pipelineEntrypoints.set(pipeline, entryPoint);
      return pipeline;
    };

    const originalSetPipeline = GPUComputePassEncoder.prototype.setPipeline;
    const computePassEntryPoints = new WeakMap(); // Store pipeline -> entrypoint mapping

    GPUComputePassEncoder.prototype.setPipeline = function (pipeline) {
      let entrypoint = pipelineEntrypoints.get(pipeline) || "unknown";
      computePassEntryPoints.set(this, entrypoint)
      return originalSetPipeline.call(this, pipeline);
    };

    const originalBeginComputePass = GPUCommandEncoder.prototype.beginComputePass;
    GPUCommandEncoder.prototype.beginComputePass = function (descriptor = {}) {
      if (!gpuDevice) {
        console.error("GPUDevice not available.");
        return originalBeginComputePass.apply(this, arguments);
      }

      // Create the query set and buffers
      const querySet = gpuDevice.createQuerySet({
        type: "timestamp",
        count: 2, // Adjust as needed for number of queries
      });
      const queryBuffer = gpuDevice.createBuffer({
        label: "Query Resolve Buffer",
        size: 2 * 8, // Enough for 2 timestamps (8 bytes each)
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });
      const stagingBuffer = gpuDevice.createBuffer({
        label: "Staging Buffer for Timestamp Results",
        size: 2 * 8, // Same size as the query buffer
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });

      const updatedDescriptor = {
        ...descriptor,
        timestampWrites: {
          querySet: querySet,
          beginningOfPassWriteIndex: 0,
          endOfPassWriteIndex: 1,
        },
      };

      const computePass = originalBeginComputePass.call(this, updatedDescriptor);

      gpuDevice.queue.onSubmittedWorkDone().then(() => {
        const commandEncoder = gpuDevice.createCommandEncoder();

        // Resolve the query set into the query buffer
        commandEncoder.resolveQuerySet(querySet, 0, 2, queryBuffer, 0);

        // Copy the query buffer results into the staging buffer
        commandEncoder.copyBufferToBuffer(queryBuffer, 0, stagingBuffer, 0, 16);

        gpuDevice.queue.submit([commandEncoder.finish()]);

        // Map and read the staging buffer
        stagingBuffer.mapAsync(GPUMapMode.READ).then(() => {
          const mappedArray = new BigUint64Array(stagingBuffer.getMappedRange());
          const [startTime, endTime] = mappedArray;
          const elapsedTimeMs = Number(endTime - startTime) / 1e6;
          
          let entrypoint = computePassEntryPoints.get(computePass) || "unknown";

          // Log entrypoint name and timing as JSON
          console.log({
            entrypoint: entrypoint,
            timeMs: elapsedTimeMs,
          });

          stagingBuffer.unmap();
          querySet.destroy();
          queryBuffer.destroy();
          stagingBuffer.destroy();

        }).catch((error) => {
          console.error("Error mapping staging buffer:", error);
        });
      });

      return computePass;
    };

    console.log("WebGPU timing query hooks installed.");
  } catch (error) {
    console.error("Error installing WebGPU hooks:", error);
  }
})();
