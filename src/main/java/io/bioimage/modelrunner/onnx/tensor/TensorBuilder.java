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
package io.bioimage.modelrunner.onnx.tensor;

import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.tensor.Utils;

import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;

import net.imglib2.RandomAccessibleInterval;
import net.imglib2.blocks.PrimitiveBlocks;
import net.imglib2.img.Img;
import net.imglib2.type.Type;
import net.imglib2.type.numeric.integer.ByteType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Util;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;

/**
 * An Onnx {@link OnnxTensor} builder from {@link Img} and
 * {@link io.bioimage.modelrunner.tensor.Tensor} objects.
 * 
 * @author Carlos Garcia Lopez de Haro
 */
public final class TensorBuilder
{

    /**
     * Not used (Utility class).
     */
    private TensorBuilder()
    {
    }

	/**
	 * Creates {@link OnnxTensor} instance with the same size and information as the
	 * given {@link io.bioimage.modelrunner.tensor.Tensor}.
	 * 
	 * @param tensor 
	 * 	The dlmodel-runner {@link io.bioimage.modelrunner.tensor.Tensor} that is
	 * 	going to be converted into a {@link OnnxTensor} tensor
	 * @param env
	 * 	{@link OrtEnvironment} needed to create {@link OnnxTensor}
	 * @return The created {@link OnnxTensor} tensor.
	 * @throws IllegalArgumentException If the type of the {@link io.bioimage.modelrunner.tensor.Tensor}
	 * is not supported
	 */
    public static OnnxTensor build(Tensor tensor, OrtEnvironment env) throws OrtException
    {
    	return build(tensor.getData(), env);
    }
    
	/**
	 * Creates {@link OnnxTensor} instance with the same size and information as the
	 * given {@link RandomAccessibleInterval}.
	 * 
     * @param <T>
     * 	possible ImgLib2 types of the {@link RandomAccessibleInterval}
	 * @param rai 
	 * 	The dlmodel-runner {@link RandomAccessibleInterval} that is
	 * 	going to be converted into a {@link OnnxTensor} tensor
	 * @param env
	 * 	{@link OrtEnvironment} needed to create {@link OnnxTensor}
	 * @return The created {@link OnnxTensor} tensor.
     * @throws OrtException if there is any Onnx error
	 * @throws IllegalArgumentException If the type of the {@link io.bioimage.modelrunner.tensor.Tensor}
	 * is not supported
	 */
    public static <T extends Type<T>> OnnxTensor build(RandomAccessibleInterval<T> rai, OrtEnvironment env) throws OrtException
    {
    	if (Util.getTypeFromInterval(rai) instanceof ByteType) {
    		return buildByte((RandomAccessibleInterval<ByteType>) rai, env);
    	} else if (Util.getTypeFromInterval(rai) instanceof IntType) {
    		return buildInt((RandomAccessibleInterval<IntType>) rai, env);
    	} else if (Util.getTypeFromInterval(rai) instanceof FloatType) {
    		return buildFloat((RandomAccessibleInterval<FloatType>) rai, env);
    	} else if (Util.getTypeFromInterval(rai) instanceof DoubleType) {
    		return buildDouble((RandomAccessibleInterval<DoubleType>) rai, env);
    	} else {
            throw new IllegalArgumentException("The image has an unsupported type: " + Util.getTypeFromInterval(rai).getClass().toString());
    	}
    }

	/**
	 * Creates a {@link OnnxTensor} tensor of type byte from an
	 * {@link RandomAccessibleInterval} of type {@link ByteType}
	 * 
	 * @param tensor 
	 * 	The {@link RandomAccessibleInterval} to fill the tensor with.
	 * @param env
	 * 	{@link OrtEnvironment} needed to create {@link OnnxTensor}
	 * @return The {@link OnnxTensor} tensor filled with the {@link RandomAccessibleInterval} data.
	 * @throws IllegalArgumentException if the input {@link RandomAccessibleInterval} type is
	 * not compatible
	 */
    private static OnnxTensor buildByte(RandomAccessibleInterval<ByteType> tensor, OrtEnvironment env) throws OrtException
    {
		tensor = Utils.transpose(tensor);
		PrimitiveBlocks< ByteType > blocks = PrimitiveBlocks.of( tensor );
		long[] tensorShape = tensor.dimensionsAsLongArray();
		int size = 1;
		for (long ll : tensorShape) size *= ll;
		final byte[] flatArr = new byte[size];
		int[] sArr = new int[tensorShape.length];
		for (int i = 0; i < sArr.length; i ++)
			sArr[i] = (int) tensorShape[i];
		blocks.copy( new long[tensorShape.length], flatArr, sArr );
    	ByteBuffer buff = ByteBuffer.wrap(flatArr);
    	OnnxTensor ndarray = OnnxTensor.createTensor(env, buff, tensorShape);
	 	return ndarray;
    }

	/**
	 * Creates a {@link OnnxTensor} tensor of type int from an
	 * {@link RandomAccessibleInterval} of type {@link IntType}
	 * 
	 * @param tensor 
	 * 	The {@link RandomAccessibleInterval} to fill the tensor with.
	 * @param env
	 * 	{@link OrtEnvironment} needed to create {@link OnnxTensor}
	 * @return The {@link OnnxTensor} tensor filled with the {@link RandomAccessibleInterval} data.
	 * @throws IllegalArgumentException if the input {@link RandomAccessibleInterval} type is
	 * not compatible
	 */
    private static OnnxTensor buildInt(RandomAccessibleInterval<IntType> tensor, OrtEnvironment env) throws OrtException
    {
		tensor = Utils.transpose(tensor);
		PrimitiveBlocks< IntType > blocks = PrimitiveBlocks.of( tensor );
		long[] tensorShape = tensor.dimensionsAsLongArray();
		int size = 1;
		for (long ll : tensorShape) size *= ll;
		final int[] flatArr = new int[size];
		int[] sArr = new int[tensorShape.length];
		for (int i = 0; i < sArr.length; i ++)
			sArr[i] = (int) tensorShape[i];
		blocks.copy( new long[tensorShape.length], flatArr, sArr );
		IntBuffer buff = IntBuffer.wrap(flatArr);
    	OnnxTensor ndarray = OnnxTensor.createTensor(env, buff, tensorShape);
	 	return ndarray;
    }

	/**
	 * Creates a {@link OnnxTensor} tensor of type float from an
	 * {@link RandomAccessibleInterval} of type {@link FloatType}
	 * 
	 * @param tensor 
	 * 	The {@link RandomAccessibleInterval} to fill the tensor with.
	 * @param env
	 * 	{@link OrtEnvironment} needed to create {@link OnnxTensor}
	 * @return The {@link OnnxTensor} tensor filled with the {@link RandomAccessibleInterval} data.
	 * @throws IllegalArgumentException if the input {@link RandomAccessibleInterval} type is
	 * not compatible
	 */
    private static OnnxTensor buildFloat(RandomAccessibleInterval<FloatType> tensor, OrtEnvironment env) throws OrtException
    {
		tensor = Utils.transpose(tensor);
		PrimitiveBlocks< FloatType > blocks = PrimitiveBlocks.of( tensor );
		long[] tensorShape = tensor.dimensionsAsLongArray();
		int size = 1;
		for (long ll : tensorShape) size *= ll;
		final float[] flatArr = new float[size];
		int[] sArr = new int[tensorShape.length];
		for (int i = 0; i < sArr.length; i ++)
			sArr[i] = (int) tensorShape[i];
		blocks.copy( new long[tensorShape.length], flatArr, sArr );
		FloatBuffer buff = FloatBuffer.wrap(flatArr);
		OnnxTensor ndarray = OnnxTensor.createTensor(env, buff, tensorShape);
	 	return ndarray;
    }

	/**
	 * Creates a {@link OnnxTensor} tensor of type double from an
	 * {@link RandomAccessibleInterval} of type {@link DoubleType}
	 * 
	 * @param tensor 
	 * 	The {@link RandomAccessibleInterval} to fill the tensor with.
	 * @param env
	 * 	{@link OrtEnvironment} needed to create {@link OnnxTensor}
	 * @return The {@link OnnxTensor} tensor filled with the {@link RandomAccessibleInterval} data.
	 * @throws IllegalArgumentException if the input {@link RandomAccessibleInterval} type is
	 * not compatible
	 */
    private static OnnxTensor buildDouble(RandomAccessibleInterval<DoubleType> tensor,  OrtEnvironment env) throws OrtException
    {
		tensor = Utils.transpose(tensor);
		PrimitiveBlocks< DoubleType > blocks = PrimitiveBlocks.of( tensor );
		long[] tensorShape = tensor.dimensionsAsLongArray();
		int size = 1;
		for (long ll : tensorShape) size *= ll;
		final double[] flatArr = new double[size];
		int[] sArr = new int[tensorShape.length];
		for (int i = 0; i < sArr.length; i ++)
			sArr[i] = (int) tensorShape[i];
		blocks.copy( new long[tensorShape.length], flatArr, sArr );
		DoubleBuffer buff = DoubleBuffer.wrap(flatArr);
    	OnnxTensor ndarray = OnnxTensor.createTensor(env, buff, tensorShape);
	 	return ndarray;
    }
}
