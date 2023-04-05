/*-
 * #%L
 * This project complements the DL-model runner acting as the engine that works loading models 
 * 		and making inference with Java API for Onnx.
 * %%
 * Copyright (C) 2022 - 2023 Institut Pasteur and BioImage.IO developers.
 * %%
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
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
 * Class to that communicates with the dl-model runner, see 
 * @see <a href="https://github.com/bioimage-io/model-runner-java">dlmodelrunner</a>
 * to execute Onnx models with the Onnx Java API.
 * This class implements the interface {@link DeepLearningEngineInterface} to get the 
 * agnostic {@link io.bioimage.modelrunner.tensor.Tensor}, convert them into 
 * {@link OnnxTensor}, execute a Onnx Deep Learning model on them and
 * convert the results back to {@link io.bioimage.modelrunner.tensor.Tensor} to send them 
 * to the main program in an agnostic manner.
 * 
 * {@link ImgLib2Builder}. Creates ImgLib2 images for the backend
 *  of {@link io.bioimage.modelrunner.tensor.Tensor} from {@link OnnxTensor}
 * {@link TensorBuilder}. Converts {@link io.bioimage.modelrunner.tensor.Tensor} into {@link OnnxTensor}
 * 
 * @author Carlos Garcia Lopez de Haro 
 */
public class OnnxInterface implements DeepLearningEngineInterface
{

    /**
     * The loaded Onnx model
     */
	private OrtSession session;
	/**
	 * An variable needed to load Onnx models
	 */
	private OrtEnvironment env;
	/**
	 * Options used to load a Onnx model
	 */
	private OrtSession.SessionOptions opts;

	/**
	 * Constructor for the interface. It is going to be called from the 
	 * dlmodel-runner
	 */
    public OnnxInterface()
    {
    }

	/**
	 * {@inheritDoc}
	 * 
     * Load a Onnx model. 
	 */
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

	/**
	 * {@inheritDoc}
	 * 
	 * Run a Onnx model on the data provided by the {@link Tensor} input list
	 * and modifies the output list with the results obtained
	 * 
	 */
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
		fillOutputTensors(output, outputTensors);
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
	 * @param onnxTensors 
	 * 	a list of Onnx tensors output of the model. The tensors are accessed with the
	 * 	corresponding names
	 * @param outputTensors 
	 * 	list of {@link Tensor} that is going to be filled with the data from the
	 * 	output Onnx {@link OnnxTensor} of the model executed
	 * @throws RunModelException If the number of tensors expected is not the same
	 *           as the number of Tensors outputed by the model
	 */
	public static void fillOutputTensors(Result onnxTensors, List<Tensor<?>> outputTensors) throws RunModelException{
		if (onnxTensors.size() != outputTensors.size())
			throw new RunModelException(onnxTensors.size(), outputTensors.size());
		for (Tensor tt : outputTensors) {
			try {
				tt.setData(ImgLib2Builder.build(onnxTensors.get(tt.getName()).get().getValue()));
			} catch (IllegalArgumentException | OrtException e) {
				e.printStackTrace();
				throw new RunModelException("Unable to recover value of output tensor: " + tt.getName()
								+ System.lineSeparator() + e.getCause().toString());
			}
		}
	}

	/**
	 * {@inheritDoc}
	 * Close the model and all the variables needed to load it and execute it
	 * once it is not needed and set them to null
	 */
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
