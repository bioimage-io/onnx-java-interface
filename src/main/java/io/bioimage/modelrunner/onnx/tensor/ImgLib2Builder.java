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


import net.imglib2.Cursor;
import net.imglib2.img.Img;
import net.imglib2.img.ImgFactory;
import net.imglib2.img.cell.CellImgFactory;
import net.imglib2.type.Type;
import net.imglib2.type.numeric.integer.ByteType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.integer.LongType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;

/**
 * A {@link Img} builder for Onnx {@link OnnxTensor} objects.
 * Build ImgLib2 objects (backend of {@link io.bioimage.modelrunner.tensor.Tensor})
 * from Onnx {@link OnnxTensor}
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
	 * Creates a {@link Img} from a given {@link OnnxTensor} tensor
	 * 
	 * @param <T> 
	 * 	the possible ImgLib2 datatypes of the image
	 * @param tensor 
	 * 	The {@link OnnxTensor} tensor data is read from.
	 * @return The {@link Img} built from the {@link OnnxTensor} tensor.
	 * @throws IllegalArgumentException If the {@link OnnxTensor} tensor type is not supported.
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
	 * Builds a {@link Img} from a byte[] obtained from a {@link OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The byte[] tensor data is read from.
	 * @return The Img built from the tensor of type {@link ByteType}.
	 */
    private static Img<ByteType> buildFromTensorByte(byte[] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length};
    	final ImgFactory< ByteType > factory = new CellImgFactory<>( new ByteType(), 5 );
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
	 * Builds a {@link Img} from a byte[][] obtained from a {@link OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The byte[][] tensor data is read from.
	 * @return The Img built from the tensor of type {@link ByteType}.
	 */
    private static Img<ByteType> buildFromTensorByte(byte[][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length};
    	final ImgFactory< ByteType > factory = new CellImgFactory<>( new ByteType(), 5 );
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
	 * Builds a {@link Img} from a byte[][][] obtained from a {@link OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The byte[][][] tensor data is read from.
	 * @return The Img built from the tensor of type {@link ByteType}.
	 */
    private static Img<ByteType> buildFromTensorByte(byte[][][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length, tensor[1].length};
    	final ImgFactory< ByteType > factory = new CellImgFactory<>( new ByteType(), 5 );
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
	 * Builds a {@link Img} from a byte[][][][] obtained from a {@link OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The byte[][][][] tensor data is read from.
	 * @return The Img built from the tensor of type {@link ByteType}.
	 */
    private static Img<ByteType> buildFromTensorByte(byte[][][][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length, tensor[1].length, tensor[2].length};
    	final ImgFactory< ByteType > factory = new CellImgFactory<>( new ByteType(), 5 );
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
	 * Builds a {@link Img} from a byte[][][][][] obtained from a {@link OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The byte[][][][][] tensor data is read from.
	 * @return The Img built from the tensor of type {@link ByteType}.
	 */
    private static Img<ByteType> buildFromTensorByte(byte[][][][][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length, 
    			tensor[1].length, tensor[2].length, tensor[3].length};
    	final ImgFactory< ByteType > factory = new CellImgFactory<>( new ByteType(), 5 );
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
	 * Builds a {@link Img} from a int[][][][][] obtained from a {@link OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The int[][][][][] tensor data is read from.
	 * @return The Img built from the tensor of type {@link IntType}.
	 */
    private static Img<IntType> buildFromTensorInt(int[][][][][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length, 
    			tensor[1].length, tensor[2].length, tensor[3].length};
    	final ImgFactory< IntType > factory = new CellImgFactory<>( new IntType(), 5 );
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
	 * Builds a {@link Img} from a int[][][][] obtained from a {@link OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The int[][][][] tensor data is read from.
	 * @return The Img built from the tensor of type {@link IntType}.
	 */
    private static Img<IntType> buildFromTensorInt(int[][][][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length, 
    			tensor[1].length, tensor[2].length};
    	final ImgFactory< IntType > factory = new CellImgFactory<>( new IntType(), 5 );
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
	 * Builds a {@link Img} from a int[][][] obtained from a {@link OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The int[][][] tensor data is read from.
	 * @return The Img built from the tensor of type {@link IntType}.
	 */
    private static Img<IntType> buildFromTensorInt(int[][][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length, 
    			tensor[1].length};
    	final ImgFactory< IntType > factory = new CellImgFactory<>( new IntType(), 5 );
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
	 * Builds a {@link Img} from a int[][] obtained from a {@link OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The int[][] tensor data is read from.
	 * @return The Img built from the tensor of type {@link IntType}.
	 */
    private static Img<IntType> buildFromTensorInt(int[][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length};
    	final ImgFactory< IntType > factory = new CellImgFactory<>( new IntType(), 5 );
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
	 * Builds a {@link Img} from a int[] obtained from a {@link OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The int[] tensor data is read from.
	 * @return The Img built from the tensor of type {@link IntType}.
	 */
    private static Img<IntType> buildFromTensorInt(int[] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length};
    	final ImgFactory< IntType > factory = new CellImgFactory<>( new IntType(), 5 );
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
	 * Builds a {@link Img} from a float[][][][][] obtained from a {@link OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The float[][][][][] tensor data is read from.
	 * @return The Img built from the tensor of type {@link FloatType}.
	 */
    private static Img<FloatType> buildFromTensorFloat(float[][][][][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length, 
    			tensor[1].length, tensor[2].length, tensor[3].length};
    	final ImgFactory< FloatType > factory = new CellImgFactory<>( new FloatType(), 5 );
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
	 * Builds a {@link Img} from a float[][][][] obtained from a {@link OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The float[][][][] tensor data is read from.
	 * @return The Img built from the tensor of type {@link FloatType}.
	 */
    private static Img<FloatType> buildFromTensorFloat(float[][][][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length, 
    			tensor[1].length, tensor[2].length};
    	final ImgFactory< FloatType > factory = new CellImgFactory<>( new FloatType(), 5 );
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
	 * Builds a {@link Img} from a float[][][] obtained from a {@link OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The float[][][] tensor data is read from.
	 * @return The Img built from the tensor of type {@link FloatType}.
	 */
    private static Img<FloatType> buildFromTensorFloat(float[][][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length, 
    			tensor[1].length};
    	final ImgFactory< FloatType > factory = new CellImgFactory<>( new FloatType(), 5 );
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
	 * Builds a {@link Img} from a float[][] obtained from a {@link OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The float[][] tensor data is read from.
	 * @return The Img built from the tensor of type {@link FloatType}.
	 */
    private static Img<FloatType> buildFromTensorFloat(float[][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length};
    	final ImgFactory< FloatType > factory = new CellImgFactory<>( new FloatType(), 5 );
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
	 * Builds a {@link Img} from a float[] obtained from a {@link OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The float[] tensor data is read from.
	 * @return The Img built from the tensor of type {@link FloatType}.
	 */
    private static Img<FloatType> buildFromTensorFloat(float[] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length};
    	final ImgFactory< FloatType > factory = new CellImgFactory<>( new FloatType(), 5 );
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
	 * Builds a {@link Img} from a double[][][][][] obtained from a {@link OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The double[][][][][] tensor data is read from.
	 * @return The Img built from the tensor of type {@link DoubleType}.
	 */
    private static Img<DoubleType> buildFromTensorDouble(double[][][][][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length, 
    			tensor[1].length, tensor[2].length, tensor[3].length};
    	final ImgFactory< DoubleType > factory = new CellImgFactory<>( new DoubleType(), 5 );
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
	 * Builds a {@link Img} from a double[][][][] obtained from a {@link OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The double[][][][] tensor data is read from.
	 * @return The Img built from the tensor of type {@link DoubleType}.
	 */
    private static Img<DoubleType> buildFromTensorDouble(double[][][][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length, 
    			tensor[1].length, tensor[2].length};
    	final ImgFactory< DoubleType > factory = new CellImgFactory<>( new DoubleType(), 5 );
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
	 * Builds a {@link Img} from a double[][][] obtained from a {@link OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The double[][][] tensor data is read from.
	 * @return The Img built from the tensor of type {@link DoubleType}.
	 */
    private static Img<DoubleType> buildFromTensorDouble(double[][][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length, 
    			tensor[1].length};
    	final ImgFactory< DoubleType > factory = new CellImgFactory<>( new DoubleType(), 5 );
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
	 * Builds a {@link Img} from a double[][] obtained from a {@link OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The double [][] tensor data is read from.
	 * @return The Img built from the tensor of type {@link DoubleType}.
	 */
    private static Img<DoubleType> buildFromTensorDouble(double[][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length};
    	final ImgFactory< DoubleType > factory = new CellImgFactory<>( new DoubleType(), 5 );
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
	 * Builds a {@link Img} from a double[] obtained from a {@link OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The double[] tensor data is read from.
	 * @return The Img built from the tensor of type {@link DoubleType}.
	 */
    private static Img<DoubleType> buildFromTensorDouble(double[] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length};
    	final ImgFactory< DoubleType > factory = new CellImgFactory<>( new DoubleType(), 5 );
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
	 * Builds a {@link Img} from a long[][][][][] obtained from a {@link OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The long[][][][][] tensor data is read from.
	 * @return The Img built from the tensor of type {@link LongType}.
	 */
    private static Img<LongType> buildFromTensorLong(long[][][][][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length, 
    			tensor[1].length, tensor[2].length, tensor[3].length};
    	final ImgFactory< LongType > factory = new CellImgFactory<>( new LongType(), 5 );
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
	 * Builds a {@link Img} from a long[][][][] obtained from a {@link OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The long[][][][] tensor data is read from.
	 * @return The Img built from the tensor of type {@link LongType}.
	 */
    private static Img<LongType> buildFromTensorLong(long[][][][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length, 
    			tensor[1].length, tensor[2].length};
    	final ImgFactory< LongType > factory = new CellImgFactory<>( new LongType(), 5 );
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
	 * Builds a {@link Img} from a long[][][] obtained from a {@link OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The long[][][] tensor data is read from.
	 * @return The Img built from the tensor of type {@link LongType}.
	 */
    private static Img<LongType> buildFromTensorLong(long[][][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length, 
    			tensor[1].length};
    	final ImgFactory< LongType > factory = new CellImgFactory<>( new LongType(), 5 );
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
	 * Builds a {@link Img} from a long[][] obtained from a {@link OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The long[][] tensor data is read from.
	 * @return The Img built from the tensor of type {@link LongType}.
	 */
    private static Img<LongType> buildFromTensorLong(long[][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length};
    	final ImgFactory< LongType > factory = new CellImgFactory<>( new LongType(), 5 );
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
	 * Builds a {@link Img} from a long[] obtained from a {@link OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The long[] tensor data is read from.
	 * @return The Img built from the tensor of type {@link LongType}.
	 */
    private static Img<LongType> buildFromTensorLong(long[] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length};
    	final ImgFactory< LongType > factory = new CellImgFactory<>( new LongType(), 5 );
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
