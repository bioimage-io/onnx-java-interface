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
package io.bioimage.modelrunner.onnx.tensor;


import net.imglib2.Cursor;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.type.Type;
import net.imglib2.type.numeric.integer.ByteType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.integer.LongType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;

/**
 * A {@link Img} builder for Onnx {@link ai.onnxruntime.OnnxTensor} objects.
 * Build ImgLib2 objects (backend of {@link io.bioimage.modelrunner.tensor.Tensor})
 * from Onnx {@link ai.onnxruntime.OnnxTensor}
 * 
 * @author Carlos Garcia Lopez de Haro
 */
public final class ImgLib2Builder
{

    /**
     * Not used (Utility class).
     */
    private ImgLib2Builder()
    {
    }

	/**
	 * Creates a {@link Img} from a given {@link ai.onnxruntime.OnnxTensor} tensor
	 * 
	 * @param <T> 
	 * 	the possible ImgLib2 datatypes of the image
	 * @param tensor 
	 * 	The {@link ai.onnxruntime.OnnxTensor} tensor data is read from.
	 * @return The {@link Img} built from the {@link ai.onnxruntime.OnnxTensor} tensor.
	 * @throws IllegalArgumentException If the {@link ai.onnxruntime.OnnxTensor} tensor type is not supported.
	 */
    @SuppressWarnings("unchecked")
    public static <T extends Type<T>> Img<T> build(Object tensor) throws IllegalArgumentException
    {
			// Create an Img of the same type of the tensor
    	if (tensor instanceof float[][][][][]) {
            return (Img<T>) buildFromTensorFloat((float[][][][][]) tensor);
    	} else if (tensor instanceof float[][][][]) {
            return (Img<T>) buildFromTensorFloat((float[][][][]) tensor);
    	} else if (tensor instanceof float[][][]) {
            return (Img<T>) buildFromTensorFloat((float[][][]) tensor);
    	} else if (tensor instanceof float[][]) {
            return (Img<T>) buildFromTensorFloat((float[][]) tensor);
    	} else if (tensor instanceof float[]) {
            return (Img<T>) buildFromTensorFloat((float[]) tensor);
    	} else if (tensor instanceof int[][][][][]) {
            return (Img<T>) buildFromTensorInt((int[][][][][]) tensor);
    	} else if (tensor instanceof int[][][][]) {
            return (Img<T>) buildFromTensorInt((int[][][][]) tensor);
    	} else if (tensor instanceof int[][][]) {
            return (Img<T>) buildFromTensorInt((int[][][]) tensor);
    	} else if (tensor instanceof int[][]) {
            return (Img<T>) buildFromTensorInt((int[][]) tensor);
    	} else if (tensor instanceof int[]) {
            return (Img<T>) buildFromTensorInt((int[]) tensor);
    	} else if (tensor instanceof double[][][][][]) {
            return (Img<T>) buildFromTensorDouble((double[][][][][]) tensor);
    	} else if (tensor instanceof double[][][][]) {
            return (Img<T>) buildFromTensorDouble((double[][][][]) tensor);
    	} else if (tensor instanceof double[][][]) {
            return (Img<T>) buildFromTensorDouble((double[][][]) tensor);
    	} else if (tensor instanceof double[][]) {
            return (Img<T>) buildFromTensorDouble((double[][]) tensor);
    	} else if (tensor instanceof double[]) {
            return (Img<T>) buildFromTensorDouble((double[]) tensor);
    	} else if (tensor instanceof byte[][][][][]) {
            return (Img<T>) buildFromTensorByte((byte[][][][][]) tensor);
    	} else if (tensor instanceof byte[][][][]) {
            return (Img<T>) buildFromTensorByte((byte[][][][]) tensor);
    	} else if (tensor instanceof byte[][][]) {
            return (Img<T>) buildFromTensorByte((byte[][][]) tensor);
    	} else if (tensor instanceof byte[][]) {
            return (Img<T>) buildFromTensorByte((byte[][]) tensor);
    	} else if (tensor instanceof byte[]) {
            return (Img<T>) buildFromTensorByte((byte[]) tensor);
    	} else if (tensor instanceof long[][][][][]) {
            return (Img<T>) buildFromTensorLong((long[][][][][]) tensor);
    	} else if (tensor instanceof long[][][][]) {
            return (Img<T>) buildFromTensorLong((long[][][][]) tensor);
    	} else if (tensor instanceof long[][][]) {
            return (Img<T>) buildFromTensorLong((long[][][]) tensor);
    	} else if (tensor instanceof long[][]) {
            return (Img<T>) buildFromTensorLong((long[][]) tensor);
    	} else if (tensor instanceof long[]) {
            return (Img<T>) buildFromTensorLong((long[]) tensor);
    	} else {
    		throw new IllegalArgumentException("Data type or tensor "
    				+ "dimensions (max=5) not supported by the software.");
    	}
    }

    /**
	 * Builds a {@link Img} from a byte[] obtained from a {@link ai.onnxruntime.OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The byte[] tensor data is read from.
	 * @return The Img built from the tensor of type {@link ByteType}.
	 */
    private static Img<ByteType> buildFromTensorByte(byte[] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length};
    	final ArrayImgFactory< ByteType > factory = new ArrayImgFactory<>( new ByteType() );
        final Img< ByteType > outputImg = factory.create(tensorShape);
    	Cursor<ByteType> tensorCursor= outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	byte val = tensor[(int) cursorPos[0]];
        	tensorCursor.get().set(val);
		}
	 	return outputImg;
	}

    /**
	 * Builds a {@link Img} from a byte[][] obtained from a {@link ai.onnxruntime.OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The byte[][] tensor data is read from.
	 * @return The Img built from the tensor of type {@link ByteType}.
	 */
    private static Img<ByteType> buildFromTensorByte(byte[][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length};
    	final ArrayImgFactory< ByteType > factory = new ArrayImgFactory<>( new ByteType() );
        final Img< ByteType > outputImg = factory.create(tensorShape);
    	Cursor<ByteType> tensorCursor= outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	byte val = tensor[(int) cursorPos[0]][(int) cursorPos[1]];
        	tensorCursor.get().set(val);
		}
	 	return outputImg;
	}

    /**
	 * Builds a {@link Img} from a byte[][][] obtained from a {@link ai.onnxruntime.OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The byte[][][] tensor data is read from.
	 * @return The Img built from the tensor of type {@link ByteType}.
	 */
    private static Img<ByteType> buildFromTensorByte(byte[][][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length, tensor[1].length};
    	final ArrayImgFactory< ByteType > factory = new ArrayImgFactory<>( new ByteType() );
        final Img< ByteType > outputImg = factory.create(tensorShape);
    	Cursor<ByteType> tensorCursor= outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	byte val = tensor[(int) cursorPos[0]][(int) cursorPos[1]][(int) cursorPos[2]];
        	tensorCursor.get().set(val);
		}
	 	return outputImg;
	}

    /**
	 * Builds a {@link Img} from a byte[][][][] obtained from a {@link ai.onnxruntime.OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The byte[][][][] tensor data is read from.
	 * @return The Img built from the tensor of type {@link ByteType}.
	 */
    private static Img<ByteType> buildFromTensorByte(byte[][][][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length, tensor[1].length, tensor[2].length};
    	final ArrayImgFactory< ByteType > factory = new ArrayImgFactory<>( new ByteType() );
        final Img< ByteType > outputImg = factory.create(tensorShape);
    	Cursor<ByteType> tensorCursor= outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	byte val = tensor[(int) cursorPos[0]][(int) cursorPos[1]][(int) cursorPos[2]][(int) cursorPos[3]];
        	tensorCursor.get().set(val);
		}
	 	return outputImg;
	}

    /**
	 * Builds a {@link Img} from a byte[][][][][] obtained from a {@link ai.onnxruntime.OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The byte[][][][][] tensor data is read from.
	 * @return The Img built from the tensor of type {@link ByteType}.
	 */
    private static Img<ByteType> buildFromTensorByte(byte[][][][][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length, 
    			tensor[1].length, tensor[2].length, tensor[3].length};
    	final ArrayImgFactory< ByteType > factory = new ArrayImgFactory<>( new ByteType() );
        final Img< ByteType > outputImg = factory.create(tensorShape);
    	Cursor<ByteType> tensorCursor= outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	byte val = tensor[(int) cursorPos[0]][(int) cursorPos[1]]
        			[(int) cursorPos[2]][(int) cursorPos[3]][(int) cursorPos[4]];
        	tensorCursor.get().set(val);
		}
	 	return outputImg;
	}

    /**
	 * Builds a {@link Img} from a int[][][][][] obtained from a {@link ai.onnxruntime.OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The int[][][][][] tensor data is read from.
	 * @return The Img built from the tensor of type {@link IntType}.
	 */
    private static Img<IntType> buildFromTensorInt(int[][][][][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length, 
    			tensor[1].length, tensor[2].length, tensor[3].length};
    	final ArrayImgFactory< IntType > factory = new ArrayImgFactory<>( new IntType() );
        final Img< IntType > outputImg = factory.create(tensorShape);
    	Cursor<IntType> tensorCursor= outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	int val = tensor[(int) cursorPos[0]][(int) cursorPos[1]]
        			[(int) cursorPos[2]][(int) cursorPos[3]][(int) cursorPos[4]];
        	tensorCursor.get().set(val);
		}
	 	return outputImg;
    }

    /**
	 * Builds a {@link Img} from a int[][][][] obtained from a {@link ai.onnxruntime.OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The int[][][][] tensor data is read from.
	 * @return The Img built from the tensor of type {@link IntType}.
	 */
    private static Img<IntType> buildFromTensorInt(int[][][][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length, 
    			tensor[1].length, tensor[2].length};
    	final ArrayImgFactory< IntType > factory = new ArrayImgFactory<>( new IntType() );
        final Img< IntType > outputImg = factory.create(tensorShape);
    	Cursor<IntType> tensorCursor= outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	int val = tensor[(int) cursorPos[0]][(int) cursorPos[1]]
        			[(int) cursorPos[2]][(int) cursorPos[3]];
        	tensorCursor.get().set(val);
		}
	 	return outputImg;
    }

    /**
	 * Builds a {@link Img} from a int[][][] obtained from a {@link ai.onnxruntime.OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The int[][][] tensor data is read from.
	 * @return The Img built from the tensor of type {@link IntType}.
	 */
    private static Img<IntType> buildFromTensorInt(int[][][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length, 
    			tensor[1].length};
    	final ArrayImgFactory< IntType > factory = new ArrayImgFactory<>( new IntType() );
        final Img< IntType > outputImg = factory.create(tensorShape);
    	Cursor<IntType> tensorCursor= outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	int val = tensor[(int) cursorPos[0]][(int) cursorPos[1]][(int) cursorPos[2]];
        	tensorCursor.get().set(val);
		}
	 	return outputImg;
    }

    /**
	 * Builds a {@link Img} from a int[][] obtained from a {@link ai.onnxruntime.OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The int[][] tensor data is read from.
	 * @return The Img built from the tensor of type {@link IntType}.
	 */
    private static Img<IntType> buildFromTensorInt(int[][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length};
    	final ArrayImgFactory< IntType > factory = new ArrayImgFactory<>( new IntType() );
        final Img< IntType > outputImg = factory.create(tensorShape);
    	Cursor<IntType> tensorCursor= outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	int val = tensor[(int) cursorPos[0]][(int) cursorPos[1]];
        	tensorCursor.get().set(val);
		}
	 	return outputImg;
    }

    /**
	 * Builds a {@link Img} from a int[] obtained from a {@link ai.onnxruntime.OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The int[] tensor data is read from.
	 * @return The Img built from the tensor of type {@link IntType}.
	 */
    private static Img<IntType> buildFromTensorInt(int[] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length};
    	final ArrayImgFactory< IntType > factory = new ArrayImgFactory<>( new IntType() );
        final Img< IntType > outputImg = factory.create(tensorShape);
    	Cursor<IntType> tensorCursor= outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	int val = tensor[(int) cursorPos[0]];
        	tensorCursor.get().set(val);
		}
	 	return outputImg;
    }

    /**
	 * Builds a {@link Img} from a float[][][][][] obtained from a {@link ai.onnxruntime.OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The float[][][][][] tensor data is read from.
	 * @return The Img built from the tensor of type {@link FloatType}.
	 */
    private static Img<FloatType> buildFromTensorFloat(float[][][][][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length, 
    			tensor[1].length, tensor[2].length, tensor[3].length};
    	final ArrayImgFactory< FloatType > factory = new ArrayImgFactory<>( new FloatType() );
        final Img< FloatType > outputImg = factory.create(tensorShape);
    	Cursor<FloatType> tensorCursor= outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	float val = tensor[(int) cursorPos[0]][(int) cursorPos[1]]
        			[(int) cursorPos[2]][(int) cursorPos[3]][(int) cursorPos[4]];
        	tensorCursor.get().set(val);
		}
	 	return outputImg;
    }

    /**
	 * Builds a {@link Img} from a float[][][][] obtained from a {@link ai.onnxruntime.OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The float[][][][] tensor data is read from.
	 * @return The Img built from the tensor of type {@link FloatType}.
	 */
    private static Img<FloatType> buildFromTensorFloat(float[][][][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length, 
    			tensor[1].length, tensor[2].length};
    	final ArrayImgFactory< FloatType > factory = new ArrayImgFactory<>( new FloatType() );
        final Img< FloatType > outputImg = factory.create(tensorShape);
    	Cursor<FloatType> tensorCursor= outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	float val = tensor[(int) cursorPos[0]][(int) cursorPos[1]]
        			[(int) cursorPos[2]][(int) cursorPos[3]];
        	tensorCursor.get().set(val);
		}
	 	return outputImg;
    }

    /**
	 * Builds a {@link Img} from a float[][][] obtained from a {@link ai.onnxruntime.OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The float[][][] tensor data is read from.
	 * @return The Img built from the tensor of type {@link FloatType}.
	 */
    private static Img<FloatType> buildFromTensorFloat(float[][][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length, 
    			tensor[1].length};
    	final ArrayImgFactory< FloatType > factory = new ArrayImgFactory<>( new FloatType() );
        final Img< FloatType > outputImg = factory.create(tensorShape);
    	Cursor<FloatType> tensorCursor= outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	float val = tensor[(int) cursorPos[0]][(int) cursorPos[1]]
        			[(int) cursorPos[2]];
        	tensorCursor.get().set(val);
		}
	 	return outputImg;
    }

    /**
	 * Builds a {@link Img} from a float[][] obtained from a {@link ai.onnxruntime.OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The float[][] tensor data is read from.
	 * @return The Img built from the tensor of type {@link FloatType}.
	 */
    private static Img<FloatType> buildFromTensorFloat(float[][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length};
    	final ArrayImgFactory< FloatType > factory = new ArrayImgFactory<>( new FloatType() );
        final Img< FloatType > outputImg = factory.create(tensorShape);
    	Cursor<FloatType> tensorCursor= outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	float val = tensor[(int) cursorPos[0]][(int) cursorPos[1]];
        	tensorCursor.get().set(val);
		}
	 	return outputImg;
    }

    /**
	 * Builds a {@link Img} from a float[] obtained from a {@link ai.onnxruntime.OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The float[] tensor data is read from.
	 * @return The Img built from the tensor of type {@link FloatType}.
	 */
    private static Img<FloatType> buildFromTensorFloat(float[] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length};
    	final ArrayImgFactory< FloatType > factory = new ArrayImgFactory<>( new FloatType() );
        final Img< FloatType > outputImg = factory.create(tensorShape);
    	Cursor<FloatType> tensorCursor= outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	float val = tensor[(int) cursorPos[0]];
        	tensorCursor.get().set(val);
		}
	 	return outputImg;
    }

    /**
	 * Builds a {@link Img} from a double[][][][][] obtained from a {@link ai.onnxruntime.OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The double[][][][][] tensor data is read from.
	 * @return The Img built from the tensor of type {@link DoubleType}.
	 */
    private static Img<DoubleType> buildFromTensorDouble(double[][][][][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length, 
    			tensor[1].length, tensor[2].length, tensor[3].length};
    	final ArrayImgFactory< DoubleType > factory = new ArrayImgFactory<>( new DoubleType() );
        final Img< DoubleType > outputImg = factory.create(tensorShape);
    	Cursor<DoubleType> tensorCursor= outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	double val = tensor[(int) cursorPos[0]][(int) cursorPos[1]]
        			[(int) cursorPos[2]][(int) cursorPos[3]][(int) cursorPos[4]];
        	tensorCursor.get().set(val);
		}
	 	return outputImg;
    }

    /**
	 * Builds a {@link Img} from a double[][][][] obtained from a {@link ai.onnxruntime.OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The double[][][][] tensor data is read from.
	 * @return The Img built from the tensor of type {@link DoubleType}.
	 */
    private static Img<DoubleType> buildFromTensorDouble(double[][][][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length, 
    			tensor[1].length, tensor[2].length};
    	final ArrayImgFactory< DoubleType > factory = new ArrayImgFactory<>( new DoubleType() );
        final Img< DoubleType > outputImg = factory.create(tensorShape);
    	Cursor<DoubleType> tensorCursor= outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	double val = tensor[(int) cursorPos[0]][(int) cursorPos[1]]
        			[(int) cursorPos[2]][(int) cursorPos[3]];
        	tensorCursor.get().set(val);
		}
	 	return outputImg;
    }

    /**
	 * Builds a {@link Img} from a double[][][] obtained from a {@link ai.onnxruntime.OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The double[][][] tensor data is read from.
	 * @return The Img built from the tensor of type {@link DoubleType}.
	 */
    private static Img<DoubleType> buildFromTensorDouble(double[][][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length, 
    			tensor[1].length};
    	final ArrayImgFactory< DoubleType > factory = new ArrayImgFactory<>( new DoubleType() );
        final Img< DoubleType > outputImg = factory.create(tensorShape);
    	Cursor<DoubleType> tensorCursor= outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	double val = tensor[(int) cursorPos[0]][(int) cursorPos[1]]
        			[(int) cursorPos[2]];
        	tensorCursor.get().set(val);
		}
	 	return outputImg;
    }

    /**
	 * Builds a {@link Img} from a double[][] obtained from a {@link ai.onnxruntime.OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The double [][] tensor data is read from.
	 * @return The Img built from the tensor of type {@link DoubleType}.
	 */
    private static Img<DoubleType> buildFromTensorDouble(double[][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length};
    	final ArrayImgFactory< DoubleType > factory = new ArrayImgFactory<>( new DoubleType() );
        final Img< DoubleType > outputImg = factory.create(tensorShape);
    	Cursor<DoubleType> tensorCursor= outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	double val = tensor[(int) cursorPos[0]][(int) cursorPos[1]];
        	tensorCursor.get().set(val);
		}
	 	return outputImg;
    }

    /**
	 * Builds a {@link Img} from a double[] obtained from a {@link ai.onnxruntime.OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The double[] tensor data is read from.
	 * @return The Img built from the tensor of type {@link DoubleType}.
	 */
    private static Img<DoubleType> buildFromTensorDouble(double[] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length};
    	final ArrayImgFactory< DoubleType > factory = new ArrayImgFactory<>( new DoubleType() );
        final Img< DoubleType > outputImg = factory.create(tensorShape);
    	Cursor<DoubleType> tensorCursor= outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	double val = tensor[(int) cursorPos[0]];
        	tensorCursor.get().set(val);
		}
	 	return outputImg;
    }

    /**
	 * Builds a {@link Img} from a long[][][][][] obtained from a {@link ai.onnxruntime.OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The long[][][][][] tensor data is read from.
	 * @return The Img built from the tensor of type {@link LongType}.
	 */
    private static Img<LongType> buildFromTensorLong(long[][][][][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length, 
    			tensor[1].length, tensor[2].length, tensor[3].length};
    	final ArrayImgFactory< LongType > factory = new ArrayImgFactory<>( new LongType() );
        final Img< LongType > outputImg = factory.create(tensorShape);
    	Cursor<LongType> tensorCursor= outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
			long val = tensor[(int) cursorPos[0]][(int) cursorPos[1]]
        			[(int) cursorPos[2]][(int) cursorPos[3]][(int) cursorPos[4]];
        	tensorCursor.get().set(val);
		}
	 	return outputImg;
    }

    /**
	 * Builds a {@link Img} from a long[][][][] obtained from a {@link ai.onnxruntime.OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The long[][][][] tensor data is read from.
	 * @return The Img built from the tensor of type {@link LongType}.
	 */
    private static Img<LongType> buildFromTensorLong(long[][][][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length, 
    			tensor[1].length, tensor[2].length};
    	final ArrayImgFactory< LongType > factory = new ArrayImgFactory<>( new LongType() );
        final Img< LongType > outputImg = factory.create(tensorShape);
    	Cursor<LongType> tensorCursor= outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
			long val = tensor[(int) cursorPos[0]][(int) cursorPos[1]]
        			[(int) cursorPos[2]][(int) cursorPos[3]];
        	tensorCursor.get().set(val);
		}
	 	return outputImg;
    }

    /**
	 * Builds a {@link Img} from a long[][][] obtained from a {@link ai.onnxruntime.OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The long[][][] tensor data is read from.
	 * @return The Img built from the tensor of type {@link LongType}.
	 */
    private static Img<LongType> buildFromTensorLong(long[][][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length, 
    			tensor[1].length};
    	final ArrayImgFactory< LongType > factory = new ArrayImgFactory<>( new LongType() );
        final Img< LongType > outputImg = factory.create(tensorShape);
    	Cursor<LongType> tensorCursor= outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
			long val = tensor[(int) cursorPos[0]][(int) cursorPos[1]]
        			[(int) cursorPos[2]];
        	tensorCursor.get().set(val);
		}
	 	return outputImg;
    }

    /**
	 * Builds a {@link Img} from a long[][] obtained from a {@link ai.onnxruntime.OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The long[][] tensor data is read from.
	 * @return The Img built from the tensor of type {@link LongType}.
	 */
    private static Img<LongType> buildFromTensorLong(long[][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length};
    	final ArrayImgFactory< LongType > factory = new ArrayImgFactory<>( new LongType() );
        final Img< LongType > outputImg = factory.create(tensorShape);
    	Cursor<LongType> tensorCursor= outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
			long val = tensor[(int) cursorPos[0]][(int) cursorPos[1]];
        	tensorCursor.get().set(val);
		}
	 	return outputImg;
    }

    /**
	 * Builds a {@link Img} from a long[] obtained from a {@link ai.onnxruntime.OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The long[] tensor data is read from.
	 * @return The Img built from the tensor of type {@link LongType}.
	 */
    private static Img<LongType> buildFromTensorLong(long[] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length};
    	final ArrayImgFactory< LongType > factory = new ArrayImgFactory<>( new LongType() );
        final Img< LongType > outputImg = factory.create(tensorShape);
    	Cursor<LongType> tensorCursor= outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
			long val = tensor[(int) cursorPos[0]];
        	tensorCursor.get().set(val);
		}
	 	return outputImg;
    }
}
