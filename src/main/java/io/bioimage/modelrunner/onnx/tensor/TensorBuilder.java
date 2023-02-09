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
package io.bioimage.modelrunner.onnx.tensor;

import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.utils.IndexingUtils;

import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;

import net.imglib2.Cursor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.type.Type;
import net.imglib2.type.numeric.integer.ByteType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Util;
import net.imglib2.view.IntervalView;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;

/**
 * A TensorFlow {@link Tensor} builder for {@link Img} and {@link Tensor}
 * objects.
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
		 * Creates a {@link Tensor} based on the provided {@link Tensor} and the
		 * desired dimension order for the resulting tensor.
		 * 
		 * @param tensor The Tensor to be converted.
		 * @return The tensor created from the sequence.
		 * @throws OrtException
		 * @throws IllegalArgumentException If the ndarray type is not supported.
		 */
    public static OnnxTensor build(Tensor tensor, OrtEnvironment env) throws OrtException
    {
    	return build(tensor.getData(), env);
    }

    /**
     * Creates a {@link Tensor} based on the provided {@link RandomAccessibleInterval} and the desired dimension order for the resulting tensor.
     * 
     * @param rai
     *        The NDArray to be converted.
     * @return The tensor created from the sequence.
     * @throws OrtException 
     * @throws IllegalArgumentException
     *         If the ndarray type is not supported.
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
		 * Creates a unsigned byte-typed {@link Tensor} based on the provided
		 * {@link RandomAccessibleInterval} and the desired dimension order for the
		 * resulting tensor.
		 * 
		 * @param imgTensor The image to be converted.
		 * @return The Img created from the sequence.
		 * @throws OrtException
		 * @throws IllegalArgumentException If the ndarray type is not supported.
		 */
    private static OnnxTensor buildByte(RandomAccessibleInterval<ByteType> imgTensor, OrtEnvironment env) throws OrtException
    {
    	long[] tensorShape = imgTensor.dimensionsAsLongArray();
    	Cursor<ByteType> tensorCursor;
		if (imgTensor instanceof IntervalView)
			tensorCursor = ((IntervalView<ByteType>) imgTensor).cursor();
		else if (imgTensor instanceof Img)
			tensorCursor = ((Img<ByteType>) imgTensor).cursor();
		else
			throw new IllegalArgumentException("The data of the " + Tensor.class + " has "
					+ "to be an instance of " + Img.class + " or " + IntervalView.class);
		long flatSize = 1;
		for (long dd : imgTensor.dimensionsAsLongArray()) { flatSize *= dd;}
		byte[] flatArr = new byte[(int) flatSize];
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos, tensorShape);
        	byte val = tensorCursor.get().getByte();
        	flatArr[flatPos] = val;
		}
    	ByteBuffer buff = ByteBuffer.wrap(flatArr);
    	OnnxTensor tensor = OnnxTensor.createTensor(env, buff, tensorShape);
	 	return tensor;
    }

    /**
		 * Creates a integer-typed {@link Tensor} based on the provided
		 * {@link RandomAccessibleInterval} and the desired dimension order for the
		 * resulting tensor.
		 * 
		 * @param imgTensor The image to be converted.
		 * @return The tensor created from the Img.
		 * @throws OrtException
		 * @throws IllegalArgumentException If the ndarray type is not supported.
		 */
    private static OnnxTensor buildInt(RandomAccessibleInterval<IntType> imgTensor, OrtEnvironment env) throws OrtException
    {
    	long[] tensorShape = imgTensor.dimensionsAsLongArray();
    	Cursor<IntType> tensorCursor;
		if (imgTensor instanceof IntervalView)
			tensorCursor = ((IntervalView<IntType>) imgTensor).cursor();
		else if (imgTensor instanceof Img)
			tensorCursor = ((Img<IntType>) imgTensor).cursor();
		else
			throw new IllegalArgumentException("The data of the " + Tensor.class + " has "
					+ "to be an instance of " + Img.class + " or " + IntervalView.class);
		long flatSize = 1;
		for (long dd : imgTensor.dimensionsAsLongArray()) { flatSize *= dd;}
		int[] flatArr = new int[(int) flatSize];
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos, tensorShape);
        	int val = tensorCursor.get().getInt();
        	flatArr[flatPos] = val;
		}
		IntBuffer buff = IntBuffer.wrap(flatArr);
    	OnnxTensor tensor = OnnxTensor.createTensor(env, buff, tensorShape);
	 	return tensor;
    }

    /**
		 * Creates a float-typed {@link Tensor} based on the provided
		 * {@link RandomAccessibleInterval} and the desired dimension order for the
		 * resulting tensor.
		 * 
		 * @param imgTensor The image to be converted.
		 * @return The tensor created from the Img.
		 * @throws OrtException
		 * @throws IllegalArgumentException If the ndarray type is not supported.
		 */
    private static OnnxTensor buildFloat(RandomAccessibleInterval<FloatType> imgTensor, OrtEnvironment env) throws OrtException
    {
    	long[] tensorShape = imgTensor.dimensionsAsLongArray();
    	Cursor<FloatType> tensorCursor;
		if (imgTensor instanceof IntervalView)
			tensorCursor = ((IntervalView<FloatType>) imgTensor).cursor();
		else if (imgTensor instanceof Img)
			tensorCursor = ((Img<FloatType>) imgTensor).cursor();
		else
			throw new IllegalArgumentException("The data of the " + Tensor.class + " has "
					+ "to be an instance of " + Img.class + " or " + IntervalView.class);
		long flatSize = 1;
		for (long dd : imgTensor.dimensionsAsLongArray()) { flatSize *= dd;}
		float[] flatArr = new float[(int) flatSize];
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos, tensorShape);
        	float val = tensorCursor.get().getRealFloat();
        	flatArr[flatPos] = val;
		}
		FloatBuffer buff = FloatBuffer.wrap(flatArr);
		OnnxTensor tensor = OnnxTensor.createTensor(env, buff, tensorShape);
	 	return tensor;
    }

    /**
		 * Creates a double-typed {@link Tensor} based on the provided
		 * {@link RandomAccessibleInterval} and the desired dimension order for the
		 * resulting tensor.
		 * 
		 * @param imgTensor The img to be converted.
		 * @return The tensor created from the Img.
		 * @throws OrtException
		 * @throws IllegalArgumentException If the ndarray type is not supported.
		 */
    private static OnnxTensor buildDouble(RandomAccessibleInterval<DoubleType> imgTensor,  OrtEnvironment env) throws OrtException
    {
    	long[] tensorShape = imgTensor.dimensionsAsLongArray();
    	Cursor<DoubleType> tensorCursor;
		if (imgTensor instanceof IntervalView)
			tensorCursor = ((IntervalView<DoubleType>) imgTensor).cursor();
		else if (imgTensor instanceof Img)
			tensorCursor = ((Img<DoubleType>) imgTensor).cursor();
		else
			throw new IllegalArgumentException("The data of the " + Tensor.class + " has "
					+ "to be an instance of " + Img.class + " or " + IntervalView.class);
		long flatSize = 1;
		for (long dd : imgTensor.dimensionsAsLongArray()) { flatSize *= dd;}
		double[] flatArr = new double[(int) flatSize];
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos, tensorShape);
        	double val = tensorCursor.get().getRealFloat();
        	flatArr[flatPos] = val;
		}
		DoubleBuffer buff = DoubleBuffer.wrap(flatArr);
    	OnnxTensor tensor = OnnxTensor.createTensor(env, buff, tensorShape);
	 	return tensor;
    }
}
