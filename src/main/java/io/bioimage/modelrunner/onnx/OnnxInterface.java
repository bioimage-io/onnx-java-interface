/*-
 * #%L
 * This project complements the DL-model runner acting as the engine that works loading models 
 * 		and making inference with Java API for Onnx.
 * %%
 * Copyright (C) 2022 - 2023 Institut Pasteur and BioImage.IO developers.
 * %%
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 * 3. Neither the name of the BioImage.io nor the names of its contributors
 *    may be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 * #L%
 */
package io.bioimage.modelrunner.onnx;

import io.bioimage.modelrunner.engine.DeepLearningEngineInterface;
import io.bioimage.modelrunner.exceptions.LoadModelException;
import io.bioimage.modelrunner.exceptions.RunModelException;
import io.bioimage.modelrunner.onnx.tensor.ImgLib2Builder;
import io.bioimage.modelrunner.onnx.tensor.TensorBuilder;
import io.bioimage.modelrunner.tensor.Tensor;

import java.util.HashMap;
import java.util.List;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.Result;
import ai.onnxruntime.OrtSession.SessionOptions;
import ai.onnxruntime.OrtSession.SessionOptions.OptLevel;


/**
 * This plugin includes the libraries to convert back and forth Onnx to ImgLib2 Tensors.
 * 
 * @see ImgLib2Builder ImgLib2Builder: Create ImgLib2 Tensors from tensors.
 * @see TensorBuilder TensorBuilder: Create tensors from ImgLib2 Tensors.
 * @author Carlos Garcia Lopez de Haro 
 */
public class OnnxInterface implements DeepLearningEngineInterface
{

    /**
     * The loaded Onnx model
     */
	private OrtSession session;
	private OrtEnvironment env;
	private OrtSession.SessionOptions opts;
	
    public OnnxInterface()
    {
    }

	@Override
	public void loadModel(String modelFolder, String modelSource) throws LoadModelException {
		try {
			env = OrtEnvironment.getEnvironment();
			opts = new SessionOptions();
			opts.setOptimizationLevel(OptLevel.BASIC_OPT);
			session = env.createSession(modelSource, opts);
		} catch (OrtException e) {
			closeModel();
			throw new LoadModelException("Error loading Onnx model", e.getCause().toString());
		} catch (Exception e) {
			closeModel();
			throw new LoadModelException("Error loading Onnx model", e.getCause().toString());
		}
		
	}

	@Override
	public void run(List<Tensor<?>> inputTensors, List<Tensor<?>> outputTensors) throws RunModelException {
		Result output;
		HashMap<String, OnnxTensor> inputMap = new HashMap<String, OnnxTensor>();
		try {
	        for (Tensor tt : inputTensors) {
	        	OnnxTensor inT = TensorBuilder.build(tt, env);
	        	inputMap.put(tt.getName(), inT);
	        }
	        output = session.run(inputMap);
		} catch (OrtException ex) {
			ex.printStackTrace();
			for (OnnxTensor tt : inputMap.values()) {
				tt.close();
			}
			throw new RunModelException("Error trying to run an Onnx model."
					+ System.lineSeparator() + ex.getCause().toString());
		}
        
		// Fill the agnostic output tensors list with data from the inference result
		outputTensors = fillOutputTensors(output, outputTensors);
		for (OnnxTensor tt : inputMap.values()) {
			tt.close();
		}
		for (Object tt : output) {
			tt = null;
		}
		output.close();
	}
	
	/**
	 * Create the list a list of output tensors agnostic to the Deep Learning
	 * engine that can be readable by Deep Icy
	 * 
	 * @param outputNDArrays an NDList containing NDArrays (tensors)
	 * @param outputTensors the names given to the tensors by the model
	 * @return a list with Deep Learning framework agnostic tensors
	 * @throws RunModelException If the number of tensors expected is not the same
	 *           as the number of Tensors outputed by the model
	 */
	public static List<Tensor<?>> fillOutputTensors(Result outputNDArrays, List<Tensor<?>> outputTensors) throws RunModelException{
		if (outputNDArrays.size() != outputTensors.size())
			throw new RunModelException(outputNDArrays.size(), outputTensors.size());
		for (Tensor tt : outputTensors) {
			try {
				tt.setData(ImgLib2Builder.build(outputNDArrays.get(tt.getName()).get().getValue()));
			} catch (IllegalArgumentException | OrtException e) {
				e.printStackTrace();
				throw new RunModelException("Unable to recover value of output tensor: " + tt.getName()
								+ System.lineSeparator() + e.getCause().toString());
			}
		}
		return outputTensors;
	}

	@Override
	public void closeModel() {
		if (env != null) {
			try {
				env.close();
				env = null;
			} catch (Exception e) {
				session = null;
			}
		}
		if (opts != null) {
			opts.close();
			opts = null;
		}
		if (session != null) {
			try {
				session.close();
				session = null;
			} catch (OrtException e) {
				session = null;
			}
		}
		
	}
}
