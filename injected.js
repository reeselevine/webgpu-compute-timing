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

    const pipelineEntryPoints = new WeakMap(); // Store pipeline -> entryPoint mapping

    // store entryPoint
    const originalCreateComputePipeline = GPUDevice.prototype.createComputePipeline;
    GPUDevice.prototype.createComputePipeline = function (descriptor) {
      const pipeline = originalCreateComputePipeline.call(this, descriptor);
      const entryPoint = descriptor?.compute?.entryPoint || "unknown";
      pipelineEntryPoints.set(pipeline, entryPoint);
      return pipeline;
    };

    // transfer entryPoint to computePass
    const originalSetPipeline = GPUComputePassEncoder.prototype.setPipeline;
    const computePassEntryPoints = new WeakMap(); // Store computePass -> entrypoint mapping
    GPUComputePassEncoder.prototype.setPipeline = function (pipeline) {
      let entrypoint = pipelineEntryPoints.get(pipeline) || "unknown";
      computePassEntryPoints.set(this, entrypoint)
      return originalSetPipeline.call(this, pipeline);
    };

    // wrap compute pass to add timestamp query
    const originalBeginComputePass = GPUCommandEncoder.prototype.beginComputePass;
    const originalEndComputePass = GPUComputePassEncoder.prototype.end;
    GPUCommandEncoder.prototype.beginComputePass = function (descriptor = {}) {
      if (!gpuDevice) {
        console.error("GPUDevice not available.");
        return originalBeginComputePass.apply(this, arguments);
      }

      if (!this.timestampQueries) {
        this.timestampQueries = [];
      }

      var timestampWrites;
      var internalTimestampWrites;
      if (descriptor.timestampWrites) {
        internalTimestampWrites = false;
        timestampWrites = descriptor.timestampWrites;
      } else {
        internalTimestampWrites = true;
        timestampWrites = {
          querySet: gpuDevice.createQuerySet({
            type: "timestamp",
            count: 2
          }),
          beginningOfPassWriteIndex: 0,
          endOfPassWriteIndex: 1
        };
      }

      // Create the query set and buffers
      const queryBuffer = gpuDevice.createBuffer({
        label: "Query Resolve Buffer",
        size: 264, // Enough for 2 timestamps (8 bytes each, each starting at a multiple of 256)
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
        //mappedAtCreation: true
      });

      // For debugging:
      //const mappedQueryBuffer = new BigInt64Array(queryBuffer.getMappedRange());
      //mappedQueryBuffer[0] = 0n;
      //mappedQueryBuffer[1] = 32n;
      //queryBuffer.unmap();

      const stagingBuffer = gpuDevice.createBuffer({
        label: "Staging Buffer for Timestamp Results",
        size: 2 * 8, // Same size as the query buffer
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        //mappedAtCreation: true
      });

      //const mappedArray = new BigInt64Array(stagingBuffer.getMappedRange());
      //mappedArray[0] = 0n;
      //mappedArray[1] = 42n;
      //stagingBuffer.unmap();


      const updatedDescriptor = {
        ...descriptor,
        timestampWrites
      };

      computePass = originalBeginComputePass.call(this, updatedDescriptor);
      computePass.end = () => {
        let entryPoint = computePassEntryPoints.get(computePass) || "unknown";
        this.timestampQueries.push({
          "timestampWrites" : timestampWrites,
          "queryBuffer": queryBuffer,
          "stagingBuffer": stagingBuffer,
          "entryPoint": entryPoint,
          "internalTimestampWrites": internalTimestampWrites
        });
        return originalEndComputePass.call(computePass);
      }
      return computePass;
    };

    // wrap command encoder finish to resolve all computePass timestamp queries.
    const originalFinish = GPUCommandEncoder.prototype.finish;
    GPUCommandEncoder.prototype.finish = function (...args) {
      // check if this command encoder has compute passes and corresponding timestamp queries
      if (this.timestampQueries != undefined) {
        // resolve all timestamp queries
        for (const timestampQuery of this.timestampQueries) {
          const timestampWrites = timestampQuery.timestampWrites;
          this.resolveQuerySet(timestampWrites.querySet, timestampWrites.beginningOfPassWriteIndex, 1, timestampQuery.queryBuffer, 0);
          this.resolveQuerySet(timestampWrites.querySet, timestampWrites.endOfPassWriteIndex, 1, timestampQuery.queryBuffer, 256);
          this.copyBufferToBuffer(timestampQuery.queryBuffer, 0, timestampQuery.stagingBuffer, 0, 8);
          this.copyBufferToBuffer(timestampQuery.queryBuffer, 256, timestampQuery.stagingBuffer, 8, 8);
        }
      }
      const commandBuffer = originalFinish.apply(this, args);
      commandBuffer.timestampQueries = this.timestampQueries;
      return commandBuffer;
    };

    // wrap queue submit to print out all timing information
    const originalSubmit = GPUQueue.prototype.submit;
    GPUQueue.prototype.submit = function(...commands) {
      const result = originalSubmit.apply(this, commands);
      for (const command of commands[0]) {
        if (command.timestampQueries != undefined) {
          for (const timestampQuery of command.timestampQueries) {
            timestampQuery.stagingBuffer.mapAsync(GPUMapMode.READ).then(() => {
              const mappedArray = new BigUint64Array(timestampQuery.stagingBuffer.getMappedRange());
              const [startTime, endTime] = mappedArray;
              const elapsedTimeMs = Number(endTime - startTime) / 1e6;
          
              // Log entrypoint name and timing as JSON
              console.log({
                entryPoint: timestampQuery.entryPoint,
                timeMs: elapsedTimeMs,
              });

              timestampQuery.stagingBuffer.unmap();
              if (timestampQuery.internalTimestampWrites) {
                timestampQuery.timestampWrites.querySet.destroy();
              }
              timestampQuery.queryBuffer.destroy();
              timestampQuery.stagingBuffer.destroy();

            }).catch((error) => {
              console.error("Error mapping staging buffer:", error);
            });
          }
        }
      }
      return result;
    }

    console.log("WebGPU timing query hooks installed.");
  } catch (error) {
    console.error("Error installing WebGPU hooks:", error);
  }
})();
