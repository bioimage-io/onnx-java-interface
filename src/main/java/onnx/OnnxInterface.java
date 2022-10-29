package onnx;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.bioimageanalysis.icy.deeplearning.exceptions.LoadModelException;
import org.bioimageanalysis.icy.deeplearning.exceptions.RunModelException;
import org.bioimageanalysis.icy.deeplearning.tensor.Tensor;
import org.bioimageanalysis.icy.deeplearning.utils.DeepLearningInterface;
import org.bioimageanalysis.icy.tensorflow.v1.tensor.ImgLib2Builder;
import org.bioimageanalysis.icy.tensorflow.v1.tensor.TensorBuilder;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.Result;
import ai.onnxruntime.OrtSession.SessionOptions;
import ai.onnxruntime.OrtSession.SessionOptions.OptLevel;


/**
 * This plugin includes the libraries to convert back and forth TensorFlow 1 to Sequences and IcyBufferedImages.
 * 
 * @see IcyBufferedImageBuilder IcyBufferedImageBuilder: Create images from tensors.
 * @see Nd4fBuilder SequenceBuilder: Create sequences from tensors.
 * @see TensorBuilder TensorBuilder: Create tensors from images and sequences.
 * @author Carlos Garcia Lopez de Haro 
 */
public class OnnxInterface implements DeepLearningInterface
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
			session = env.createSession(modelFolder, opts);
		} catch (OrtException e) {
			closeModel();
			throw new LoadModelException("Error loading Onnx model", e.getCause().toString());
		}
		
	}

	@Override
	public List<Tensor<?>> run(List<Tensor<?>> inputTensors, List<Tensor<?>> outputTensors) throws RunModelException {
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
		return outputTensors;
	}
	
	/**
	 * Create the list a list of output tensors agnostic to the Deep Learning engine
	 * that can be readable by Deep Icy
	 * @param outputTensors
	 * 	an NDList containing NDArrays (tensors)
	 * @param outputTensors2
	 * 	the names given to the tensors by the model
	 * @return a list with Deep Learning framework agnostic tensors
	 * @throws RunModelException If the number of tensors expected is not the same as the number of
	 * 	Tensors outputed by the model
	 */
	public static List<Tensor<?>> fillOutputTensors(Result outputNDArrays, List<Tensor<?>> outputTensors) throws RunModelException{
		if (outputNDArrays.size() != outputTensors.size())
			throw new RunModelException(outputNDArrays.size(), outputTensors.size());
		for (int i = 0; i < outputNDArrays.size(); i ++) {
			outputTensors.get(i).setData(ImgLib2Builder.build(outputNDArrays.get(i)));
		}
		return outputTensors;
	}

	@Override
	public void closeModel() {
		if (env != null) {
			env.close();
			env = null;
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
