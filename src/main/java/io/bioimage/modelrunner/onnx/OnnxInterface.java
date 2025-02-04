/*-
 * #%L
 * This project complements the DL-model runner acting as the engine that works loading models 
 * 		and making inference with Java API for Onnx.
 * %%
 * Copyright (C) 2022 - 2023 Institut Pasteur and BioImage.IO developers.
 * %%
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *      http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * #L%
 */
package io.bioimage.modelrunner.onnx;

import io.bioimage.modelrunner.apposed.appose.Types;
import io.bioimage.modelrunner.engine.DeepLearningEngineInterface;
import io.bioimage.modelrunner.exceptions.LoadModelException;
import io.bioimage.modelrunner.exceptions.RunModelException;
import io.bioimage.modelrunner.onnx.tensor.ImgLib2Builder;
import io.bioimage.modelrunner.onnx.tensor.TensorBuilder;
import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedHashMap;
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
    public static void main(String args[]) throws LoadModelException, RunModelException {
    	String folderName = "/home/carlos/git/deep-icy/models/NucleiSegmentationBoundaryModel_27112023_190556";
    	String source = folderName + File.separator + "weights.onnx";
    	
    	OnnxInterface oi = new OnnxInterface();
    	oi.loadModel(folderName, source);
    	
    	RandomAccessibleInterval<FloatType> img = ArrayImgs.floats(new long[] {1, 1, 256, 256});
    	Tensor<FloatType> tt = Tensor.build("input0", "bcyx", img);
    	List<Tensor<?>> inps = new ArrayList<Tensor<?>>();
    	inps.add(tt);

    	Tensor<FloatType> oo = Tensor.buildEmptyTensor("output0", "bcyx");
    	List<Tensor<?>> outs = new ArrayList<Tensor<?>>();
    	outs.add(oo);
    	
    	oi.run(inps, outs);
    	System.out.println(false);
    }
    */

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
			throw new LoadModelException("Error loading Onnx model", e.toString());
		} catch (Exception e) {
			closeModel();
			throw new LoadModelException("Error loading Onnx model", e.toString());
		}
		
	}

	/**
	 * {@inheritDoc}
	 * 
	 * Run a Onnx model on the data provided by the {@link Tensor} input list
	 * and modifies the output list with the results obtained
	 * @throws RunModelException 
	 * 
	 */
	@Override
	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
	void run(List<Tensor<T>> inputTensors, List<Tensor<R>> outputTensors) throws RunModelException {
		Result output;
		LinkedHashMap<String, OnnxTensor> inputMap = new LinkedHashMap<String, OnnxTensor>();
		Iterator<String> inputNames = session.getInputNames().iterator();
		Iterator<String> outputNames = session.getOutputNames().iterator();
		try {
	        for (Tensor tt : inputTensors) {
	        	OnnxTensor inT = TensorBuilder.build(tt, env);
	        	inputMap.put(inputNames.next(), inT);
	        }
	        output = session.run(inputMap);
		} catch (OrtException ex) {
			ex.printStackTrace();
			for (OnnxTensor tt : inputMap.values()) {
				tt.close();
			}
			throw new RunModelException("Error trying to run an Onnx model."
					+ System.lineSeparator() + ex.toString());
		}
        
		// Fill the agnostic output tensors list with data from the inference result
		fillOutputTensors(output, outputTensors);
		for (OnnxTensor tt : inputMap.values()) {
			tt.close();
		}
		output.close();
	}

	@Override
	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>> List<RandomAccessibleInterval<R>> inference(
			List<RandomAccessibleInterval<T>> inputs) throws RunModelException {
		Result output;
		LinkedHashMap<String, OnnxTensor> inputMap = new LinkedHashMap<String, OnnxTensor>();
		Iterator<String> inputNames = session.getInputNames().iterator();
		try {
	        for (RandomAccessibleInterval<T> tt : inputs) {
	        	OnnxTensor inT = TensorBuilder.build(tt, env);
	        	inputMap.put(inputNames.next(), inT);
	        }
	        output = session.run(inputMap);
		} catch (OrtException ex) {
			for (OnnxTensor tt : inputMap.values()) {
				tt.close();
			}
			throw new RunModelException("Error trying to run an Onnx model."
					+ System.lineSeparator() + Types.stackTrace(ex));
		}
		for (OnnxTensor tt : inputMap.values()) {
			tt.close();
		}
        
		// Fill the agnostic output tensors list with data from the inference result
		List<RandomAccessibleInterval<R>> rais = new ArrayList<RandomAccessibleInterval<R>>();
		for (int i = 0; i < output.size(); i ++) {
			try {
				rais.add(ImgLib2Builder.build(output.get(i).getValue()));
				output.get(i).close();
			} catch (IllegalArgumentException | OrtException e) {
				for (int j = i; j < output.size(); j ++)
					output.get(j).close();
				output.close();
				throw new RunModelException("Error converting tensor into RAI" + Types.stackTrace(e));
			}
		}
		output.close();
		return rais;
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
	public static <T extends RealType<T> & NativeType<T>>
	void fillOutputTensors(Result onnxTensors,
		List<Tensor<T>> outputTensors) throws RunModelException {
		if (onnxTensors.size() != outputTensors.size())
			throw new RunModelException(onnxTensors.size(), outputTensors.size());
		int cc = 0;
		for (Tensor tt : outputTensors) {
			try {
				tt.setData(ImgLib2Builder.build(onnxTensors.get(cc).getValue()));
				onnxTensors.get(cc).close();
				cc ++;
			} catch (IllegalArgumentException | OrtException e) {
				for (int j = cc; j < onnxTensors.size(); j ++)
					onnxTensors.get(j).close();
				onnxTensors.close();
				throw new RunModelException("Error converting tensor '" + tt.getName() + "' into RAI" + Types.stackTrace(e));
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
