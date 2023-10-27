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


import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.Type;
import net.imglib2.type.numeric.integer.ByteType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.integer.LongType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;

/**
 * A {@link RandomAccessibleInterval} builder for Onnx {@link ai.onnxruntime.OnnxTensor} objects.
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
	 * Creates a {@link RandomAccessibleInterval} from a given {@link ai.onnxruntime.OnnxTensor} tensor
	 * 
	 * @param <T> 
	 * 	the possible ImgLib2 datatypes of the image
	 * @param tensor 
	 * 	The {@link ai.onnxruntime.OnnxTensor} tensor data is read from.
	 * @return The {@link RandomAccessibleInterval} built from the {@link ai.onnxruntime.OnnxTensor} tensor.
	 * @throws IllegalArgumentException If the {@link ai.onnxruntime.OnnxTensor} tensor type is not supported.
	 */
    @SuppressWarnings("unchecked")
    public static <T extends Type<T>> RandomAccessibleInterval<T> build(Object tensor) throws IllegalArgumentException
    {
			// Create an Img of the same type of the tensor
    	if (tensor instanceof float[][][][][]) {
            return (RandomAccessibleInterval<T>) buildFromTensorFloat((float[][][][][]) tensor);
    	} else if (tensor instanceof float[][][][]) {
            return (RandomAccessibleInterval<T>) buildFromTensorFloat((float[][][][]) tensor);
    	} else if (tensor instanceof float[][][]) {
            return (RandomAccessibleInterval<T>) buildFromTensorFloat((float[][][]) tensor);
    	} else if (tensor instanceof float[][]) {
            return (RandomAccessibleInterval<T>) buildFromTensorFloat((float[][]) tensor);
    	} else if (tensor instanceof float[]) {
            return (RandomAccessibleInterval<T>) buildFromTensorFloat((float[]) tensor);
    	} else if (tensor instanceof int[][][][][]) {
            return (RandomAccessibleInterval<T>) buildFromTensorInt((int[][][][][]) tensor);
    	} else if (tensor instanceof int[][][][]) {
            return (RandomAccessibleInterval<T>) buildFromTensorInt((int[][][][]) tensor);
    	} else if (tensor instanceof int[][][]) {
            return (RandomAccessibleInterval<T>) buildFromTensorInt((int[][][]) tensor);
    	} else if (tensor instanceof int[][]) {
            return (RandomAccessibleInterval<T>) buildFromTensorInt((int[][]) tensor);
    	} else if (tensor instanceof int[]) {
            return (RandomAccessibleInterval<T>) buildFromTensorInt((int[]) tensor);
    	} else if (tensor instanceof double[][][][][]) {
            return (RandomAccessibleInterval<T>) buildFromTensorDouble((double[][][][][]) tensor);
    	} else if (tensor instanceof double[][][][]) {
            return (RandomAccessibleInterval<T>) buildFromTensorDouble((double[][][][]) tensor);
    	} else if (tensor instanceof double[][][]) {
            return (RandomAccessibleInterval<T>) buildFromTensorDouble((double[][][]) tensor);
    	} else if (tensor instanceof double[][]) {
            return (RandomAccessibleInterval<T>) buildFromTensorDouble((double[][]) tensor);
    	} else if (tensor instanceof double[]) {
            return (RandomAccessibleInterval<T>) buildFromTensorDouble((double[]) tensor);
    	} else if (tensor instanceof byte[][][][][]) {
            return (RandomAccessibleInterval<T>) buildFromTensorByte((byte[][][][][]) tensor);
    	} else if (tensor instanceof byte[][][][]) {
            return (RandomAccessibleInterval<T>) buildFromTensorByte((byte[][][][]) tensor);
    	} else if (tensor instanceof byte[][][]) {
            return (RandomAccessibleInterval<T>) buildFromTensorByte((byte[][][]) tensor);
    	} else if (tensor instanceof byte[][]) {
            return (RandomAccessibleInterval<T>) buildFromTensorByte((byte[][]) tensor);
    	} else if (tensor instanceof byte[]) {
            return (RandomAccessibleInterval<T>) buildFromTensorByte((byte[]) tensor);
    	} else if (tensor instanceof long[][][][][]) {
            return (RandomAccessibleInterval<T>) buildFromTensorLong((long[][][][][]) tensor);
    	} else if (tensor instanceof long[][][][]) {
            return (RandomAccessibleInterval<T>) buildFromTensorLong((long[][][][]) tensor);
    	} else if (tensor instanceof long[][][]) {
            return (RandomAccessibleInterval<T>) buildFromTensorLong((long[][][]) tensor);
    	} else if (tensor instanceof long[][]) {
            return (RandomAccessibleInterval<T>) buildFromTensorLong((long[][]) tensor);
    	} else if (tensor instanceof long[]) {
            return (RandomAccessibleInterval<T>) buildFromTensorLong((long[]) tensor);
    	} else {
    		throw new IllegalArgumentException("Data type or tensor "
    				+ "dimensions (max=5) not supported by the software.");
    	}
    }

    /**
	 * Builds a {@link RandomAccessibleInterval} from a byte[] obtained from a {@link ai.onnxruntime.OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The byte[] tensor data is read from.
	 * @return The RandomAccessibleInterval built from the tensor of type {@link ByteType}.
	 */
    private static RandomAccessibleInterval<ByteType> buildFromTensorByte(byte[] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length};
		int totalSize = 1;
		for (long i : tensorShape) totalSize *= i;
		byte[] flatArr = new byte[totalSize];
		int cc = 0;
		for (int i = 0; i < tensor.length; i ++) {
			flatArr[cc ++] = tensor[i];
		}
		RandomAccessibleInterval<ByteType> rai = ArrayImgs.bytes(flatArr, tensorShape);
		return rai;
	}

    /**
	 * Builds a {@link RandomAccessibleInterval} from a byte[][] obtained from a {@link ai.onnxruntime.OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The byte[][] tensor data is read from.
	 * @return The RandomAccessibleInterval built from the tensor of type {@link ByteType}.
	 */
    private static RandomAccessibleInterval<ByteType> buildFromTensorByte(byte[][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length};
		int totalSize = 1;
		for (long i : tensorShape) totalSize *= i;
		byte[] flatArr = new byte[totalSize];
		int cc = 0;
		for (int j = 0; j < tensor[0].length; j ++) {
			for (int i = 0; i < tensor.length; i ++) {
				flatArr[cc ++] = tensor[i][j];
			}
		}
		RandomAccessibleInterval<ByteType> rai = ArrayImgs.bytes(flatArr, tensorShape);
		return rai;
	}

    /**
	 * Builds a {@link RandomAccessibleInterval} from a byte[][][] obtained from a {@link ai.onnxruntime.OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The byte[][][] tensor data is read from.
	 * @return The Img built from the tensor of type {@link ByteType}.
	 */
    private static RandomAccessibleInterval<ByteType> buildFromTensorByte(byte[][][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length, tensor[0][0].length};
		int totalSize = 1;
		for (long i : tensorShape) totalSize *= i;
		byte[] flatArr = new byte[totalSize];
		int cc = 0;
		for (int k = 0; k < tensor[0][0].length; k ++) {
			for (int j = 0; j < tensor[0].length; j ++) {
				for (int i = 0; i < tensor.length; i ++) {
					flatArr[cc ++] = tensor[i][j][k];
				}
			}
		}
		RandomAccessibleInterval<ByteType> rai = ArrayImgs.bytes(flatArr, tensorShape);
		return rai;
	}

    /**
	 * Builds a {@link RandomAccessibleInterval} from a byte[][][][] obtained from a {@link ai.onnxruntime.OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The byte[][][][] tensor data is read from.
	 * @return The RandomAccessibleInterval built from the tensor of type {@link ByteType}.
	 */
    private static RandomAccessibleInterval<ByteType> buildFromTensorByte(byte[][][][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length, 
    			tensor[0][0].length, tensor[0][0][0].length};
		int totalSize = 1;
		for (long i : tensorShape) totalSize *= i;
		byte[] flatArr = new byte[totalSize];
		int cc = 0;
		for (int l = 0; l < tensor[0][0][0].length; l ++) {
			for (int k = 0; k < tensor[0][0].length; k ++) {
				for (int j = 0; j < tensor[0].length; j ++) {
					for (int i = 0; i < tensor.length; i ++) {
						flatArr[cc ++] = tensor[i][j][k][l];
					}
				}
			}
		}
		RandomAccessibleInterval<ByteType> rai = ArrayImgs.bytes(flatArr, tensorShape);
		return rai;
	}

    /**
	 * Builds a {@link RandomAccessibleInterval} from a byte[][][][][] obtained from a {@link ai.onnxruntime.OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The byte[][][][][] tensor data is read from.
	 * @return The RandomAccessibleInterval built from the tensor of type {@link ByteType}.
	 */
    private static RandomAccessibleInterval<ByteType> buildFromTensorByte(byte[][][][][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length, 
    			tensor[0][0].length, tensor[0][0][0].length, tensor[0][0][0][0].length};
		int totalSize = 1;
		for (long i : tensorShape) totalSize *= i;
		byte[] flatArr = new byte[totalSize];
		int cc = 0;
		for (int m = 0; m < tensor[0][0][0][0].length; m ++) {
			for (int l = 0; l < tensor[0][0][0].length; l ++) {
				for (int k = 0; k < tensor[0][0].length; k ++) {
					for (int j = 0; j < tensor[0].length; j ++) {
						for (int i = 0; i < tensor.length; i ++) {
							flatArr[cc ++] = tensor[i][j][k][l][m];
						}
					}
				}
			}
		}
		RandomAccessibleInterval<ByteType> rai = ArrayImgs.bytes(flatArr, tensorShape);
		return rai;
	}

    /**
	 * Builds a {@link RandomAccessibleInterval} from a int[][][][][] obtained from a {@link ai.onnxruntime.OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The int[][][][][] tensor data is read from.
	 * @return The RandomAccessibleInterval built from the tensor of type {@link IntType}.
	 */
    private static RandomAccessibleInterval<IntType> buildFromTensorInt(int[][][][][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length, 
    			tensor[0][0].length, tensor[0][0][0].length, tensor[0][0][0][0].length};
		int totalSize = 1;
		for (long i : tensorShape) totalSize *= i;
		int[] flatArr = new int[totalSize];
		int cc = 0;
		for (int m = 0; m < tensor[0][0][0][0].length; m ++) {
			for (int l = 0; l < tensor[0][0][0].length; l ++) {
				for (int k = 0; k < tensor[0][0].length; k ++) {
					for (int j = 0; j < tensor[0].length; j ++) {
						for (int i = 0; i < tensor.length; i ++) {
							flatArr[cc ++] = tensor[i][j][k][l][m];
						}
					}
				}
			}
		}
		RandomAccessibleInterval<IntType> rai = ArrayImgs.ints(flatArr, tensorShape);
		return rai;
    }

    /**
	 * Builds a {@link RandomAccessibleInterval} from a int[][][][] obtained from a {@link ai.onnxruntime.OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The int[][][][] tensor data is read from.
	 * @return The RandomAccessibleInterval built from the tensor of type {@link IntType}.
	 */
    private static RandomAccessibleInterval<IntType> buildFromTensorInt(int[][][][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length, 
    			tensor[0][0].length, tensor[0][0][0].length};
		int totalSize = 1;
		for (long i : tensorShape) totalSize *= i;
		int[] flatArr = new int[totalSize];
		int cc = 0;
		for (int l = 0; l < tensor[0][0][0].length; l ++) {
			for (int k = 0; k < tensor[0][0].length; k ++) {
				for (int j = 0; j < tensor[0].length; j ++) {
					for (int i = 0; i < tensor.length; i ++) {
						flatArr[cc ++] = tensor[i][j][k][l];
					}
				}
			}
		}
		RandomAccessibleInterval<IntType> rai = ArrayImgs.ints(flatArr, tensorShape);
		return rai;
    }

    /**
	 * Builds a {@link RandomAccessibleInterval} from a int[][][] obtained from a {@link ai.onnxruntime.OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The int[][][] tensor data is read from.
	 * @return The RandomAccessibleInterval built from the tensor of type {@link IntType}.
	 */
    private static RandomAccessibleInterval<IntType> buildFromTensorInt(int[][][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length, 
    			tensor[0][0].length};
		int totalSize = 1;
		for (long i : tensorShape) totalSize *= i;
		int[] flatArr = new int[totalSize];
		int cc = 0;
		for (int k = 0; k < tensor[0][0].length; k ++) {
			for (int j = 0; j < tensor[0].length; j ++) {
				for (int i = 0; i < tensor.length; i ++) {
					flatArr[cc ++] = tensor[i][j][k];
				}
			}
		}
		RandomAccessibleInterval<IntType> rai = ArrayImgs.ints(flatArr, tensorShape);
		return rai;
    }

    /**
	 * Builds a {@link RandomAccessibleInterval} from a int[][] obtained from a {@link ai.onnxruntime.OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The int[][] tensor data is read from.
	 * @return The RandomAccessibleInterval built from the tensor of type {@link IntType}.
	 */
    private static RandomAccessibleInterval<IntType> buildFromTensorInt(int[][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length};
		int totalSize = 1;
		for (long i : tensorShape) totalSize *= i;
		int[] flatArr = new int[totalSize];
		int cc = 0;
		for (int j = 0; j < tensor[0].length; j ++) {
			for (int i = 0; i < tensor.length; i ++) {
				flatArr[cc ++] = tensor[i][j];
			}
		}
		RandomAccessibleInterval<IntType> rai = ArrayImgs.ints(flatArr, tensorShape);
		return rai;
    }

    /**
	 * Builds a {@link RandomAccessibleInterval} from a int[] obtained from a {@link ai.onnxruntime.OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The int[] tensor data is read from.
	 * @return The RandomAccessibleInterval built from the tensor of type {@link IntType}.
	 */
    private static RandomAccessibleInterval<IntType> buildFromTensorInt(int[] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length};
		RandomAccessibleInterval<IntType> rai = ArrayImgs.ints(tensor, tensorShape);
		return rai;
    }

    /**
	 * Builds a {@link RandomAccessibleInterval} from a float[][][][][] obtained from a {@link ai.onnxruntime.OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The float[][][][][] tensor data is read from.
	 * @return The RandomAccessibleInterval built from the tensor of type {@link FloatType}.
	 */
    private static RandomAccessibleInterval<FloatType> buildFromTensorFloat(float[][][][][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length, 
    			tensor[0][0].length, tensor[0][0][0].length, tensor[0][0][0][0].length};
		int totalSize = 1;
		for (long i : tensorShape) totalSize *= i;
		float[] flatArr = new float[totalSize];
		int cc = 0;
		for (int m = 0; m < tensor[0][0][0][0].length; m ++) {
			for (int l = 0; l < tensor[0][0][0].length; l ++) {
				for (int k = 0; k < tensor[0][0].length; k ++) {
					for (int j = 0; j < tensor[0].length; j ++) {
						for (int i = 0; i < tensor.length; i ++) {
							flatArr[cc ++] = tensor[i][j][k][l][m];
						}
					}
				}
			}
		}
		RandomAccessibleInterval<FloatType> rai = ArrayImgs.floats(flatArr, tensorShape);
		return rai;
    }

    /**
	 * Builds a {@link RandomAccessibleInterval} from a float[][][][] obtained from a {@link ai.onnxruntime.OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The float[][][][] tensor data is read from.
	 * @return The RandomAccessibleInterval built from the tensor of type {@link FloatType}.
	 */
    private static RandomAccessibleInterval<FloatType> buildFromTensorFloat(float[][][][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length, 
    			tensor[0][0].length, tensor[0][0][0].length};
		int totalSize = 1;
		for (long i : tensorShape) totalSize *= i;
		float[] flatArr = new float[totalSize];
		int cc = 0;
		for (int l = 0; l < tensor[0][0][0].length; l ++) {
			for (int k = 0; k < tensor[0][0].length; k ++) {
				for (int j = 0; j < tensor[0].length; j ++) {
					for (int i = 0; i < tensor.length; i ++) {
						flatArr[cc ++] = tensor[i][j][k][l];
					}
				}
			}
		}
		RandomAccessibleInterval<FloatType> rai = ArrayImgs.floats(flatArr, tensorShape);
		return rai;
    }

    /**
	 * Builds a {@link RandomAccessibleInterval} from a float[][][] obtained from a {@link ai.onnxruntime.OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The float[][][] tensor data is read from.
	 * @return The RandomAccessibleInterval built from the tensor of type {@link FloatType}.
	 */
    private static RandomAccessibleInterval<FloatType> buildFromTensorFloat(float[][][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length, 
    			tensor[0][0].length};
		int totalSize = 1;
		for (long i : tensorShape) totalSize *= i;
		float[] flatArr = new float[totalSize];
		int cc = 0;
		for (int k = 0; k < tensor[0][0].length; k ++) {
			for (int j = 0; j < tensor[0].length; j ++) {
				for (int i = 0; i < tensor.length; i ++) {
					flatArr[cc ++] = tensor[i][j][k];
				}
			}
		}
		RandomAccessibleInterval<FloatType> rai = ArrayImgs.floats(flatArr, tensorShape);
		return rai;
    }

    /**
	 * Builds a {@link RandomAccessibleInterval} from a float[][] obtained from a {@link ai.onnxruntime.OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The float[][] tensor data is read from.
	 * @return The RandomAccessibleInterval built from the tensor of type {@link FloatType}.
	 */
    private static RandomAccessibleInterval<FloatType> buildFromTensorFloat(float[][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length};
		int totalSize = 1;
		for (long i : tensorShape) totalSize *= i;
		float[] flatArr = new float[totalSize];
		int cc = 0;
		for (int j = 0; j < tensor[0].length; j ++) {
			for (int i = 0; i < tensor.length; i ++) {
				flatArr[cc ++] = tensor[i][j];
			}
		}
		RandomAccessibleInterval<FloatType> rai = ArrayImgs.floats(flatArr, tensorShape);
		return rai;
    }

    /**
	 * Builds a {@link RandomAccessibleInterval} from a float[] obtained from a {@link ai.onnxruntime.OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The float[] tensor data is read from.
	 * @return The RandomAccessibleInterval built from the tensor of type {@link FloatType}.
	 */
    private static RandomAccessibleInterval<FloatType> buildFromTensorFloat(float[] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length};
		RandomAccessibleInterval<FloatType> rai = ArrayImgs.floats(tensor, tensorShape);
		return rai;
    }

    /**
	 * Builds a {@link RandomAccessibleInterval} from a double[][][][][] obtained from a {@link ai.onnxruntime.OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The double[][][][][] tensor data is read from.
	 * @return The RandomAccessibleInterval built from the tensor of type {@link DoubleType}.
	 */
    private static RandomAccessibleInterval<DoubleType> buildFromTensorDouble(double[][][][][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length, 
    			tensor[0][0].length, tensor[0][0][0].length, tensor[0][0][0][0].length};
		int totalSize = 1;
		for (long i : tensorShape) totalSize *= i;
		double[] flatArr = new double[totalSize];
		int cc = 0;
		for (int m = 0; m < tensor[0][0][0][0].length; m ++) {
			for (int l = 0; l < tensor[0][0][0].length; l ++) {
				for (int k = 0; k < tensor[0][0].length; k ++) {
					for (int j = 0; j < tensor[0].length; j ++) {
						for (int i = 0; i < tensor.length; i ++) {
							flatArr[cc ++] = tensor[i][j][k][l][m];
						}
					}
				}
			}
		}
		RandomAccessibleInterval<DoubleType> rai = ArrayImgs.doubles(flatArr, tensorShape);
		return rai;
    }

    /**
	 * Builds a {@link RandomAccessibleInterval} from a double[][][][] obtained from a {@link ai.onnxruntime.OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The double[][][][] tensor data is read from.
	 * @return The RandomAccessibleInterval built from the tensor of type {@link DoubleType}.
	 */
    private static RandomAccessibleInterval<DoubleType> buildFromTensorDouble(double[][][][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length, 
    			tensor[0][0].length, tensor[0][0][0].length};
		int totalSize = 1;
		for (long i : tensorShape) totalSize *= i;
		double[] flatArr = new double[totalSize];
		int cc = 0;
		for (int l = 0; l < tensor[0][0][0].length; l ++) {
			for (int k = 0; k < tensor[0][0].length; k ++) {
				for (int j = 0; j < tensor[0].length; j ++) {
					for (int i = 0; i < tensor.length; i ++) {
						flatArr[cc ++] = tensor[i][j][k][l];
					}
				}
			}
		}
		RandomAccessibleInterval<DoubleType> rai = ArrayImgs.doubles(flatArr, tensorShape);
		return rai;
    }

    /**
	 * Builds a {@link RandomAccessibleInterval} from a double[][][] obtained from a {@link ai.onnxruntime.OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The double[][][] tensor data is read from.
	 * @return The RandomAccessibleInterval built from the tensor of type {@link DoubleType}.
	 */
    private static RandomAccessibleInterval<DoubleType> buildFromTensorDouble(double[][][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length, 
    			tensor[0][0].length};
		int totalSize = 1;
		for (long i : tensorShape) totalSize *= i;
		double[] flatArr = new double[totalSize];
		int cc = 0;
		for (int k = 0; k < tensor[0][0].length; k ++) {
			for (int j = 0; j < tensor[0].length; j ++) {
				for (int i = 0; i < tensor.length; i ++) {
					flatArr[cc ++] = tensor[i][j][k];
				}
			}
		}
		RandomAccessibleInterval<DoubleType> rai = ArrayImgs.doubles(flatArr, tensorShape);
		return rai;
    }

    /**
	 * Builds a {@link RandomAccessibleInterval} from a double[][] obtained from a {@link ai.onnxruntime.OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The double [][] tensor data is read from.
	 * @return The RandomAccessibleInterval built from the tensor of type {@link DoubleType}.
	 */
    private static RandomAccessibleInterval<DoubleType> buildFromTensorDouble(double[][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length};
		int totalSize = 1;
		for (long i : tensorShape) totalSize *= i;
		double[] flatArr = new double[totalSize];
		int cc = 0;
		for (int j = 0; j < tensor[0].length; j ++) {
			for (int i = 0; i < tensor.length; i ++) {
				flatArr[cc ++] = tensor[i][j];
			}
		}
		RandomAccessibleInterval<DoubleType> rai = ArrayImgs.doubles(flatArr, tensorShape);
		return rai;
    }

    /**
	 * Builds a {@link RandomAccessibleInterval} from a double[] obtained from a {@link ai.onnxruntime.OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The double[] tensor data is read from.
	 * @return The RandomAccessibleInterval built from the tensor of type {@link DoubleType}.
	 */
    private static RandomAccessibleInterval<DoubleType> buildFromTensorDouble(double[] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length};
		int totalSize = 1;
		for (long i : tensorShape) totalSize *= i;
		double[] flatArr = new double[totalSize];
		int cc = 0;
		for (int i = 0; i < tensor.length; i ++) {
			flatArr[cc ++] = tensor[i];
		}
		RandomAccessibleInterval<DoubleType> rai = ArrayImgs.doubles(flatArr, tensorShape);
		return rai;
    }

    /**
	 * Builds a {@link RandomAccessibleInterval} from a long[][][][][] obtained from a {@link ai.onnxruntime.OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The long[][][][][] tensor data is read from.
	 * @return The RandomAccessibleInterval built from the tensor of type {@link LongType}.
	 */
    private static RandomAccessibleInterval<LongType> buildFromTensorLong(long[][][][][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length, 
    			tensor[0][0].length, tensor[0][0][0].length, tensor[0][0][0][0].length};
		int totalSize = 1;
		for (long i : tensorShape) totalSize *= i;
		long[] flatArr = new long[totalSize];
		int cc = 0;
		for (int m = 0; m < tensor[0][0][0][0].length; m ++) {
			for (int l = 0; l < tensor[0][0][0].length; l ++) {
				for (int k = 0; k < tensor[0][0].length; k ++) {
					for (int j = 0; j < tensor[0].length; j ++) {
						for (int i = 0; i < tensor.length; i ++) {
							flatArr[cc ++] = tensor[i][j][k][l][m];
						}
					}
				}
			}
		}
		RandomAccessibleInterval<LongType> rai = ArrayImgs.longs(flatArr, tensorShape);
		return rai;
    }

    /**
	 * Builds a {@link RandomAccessibleInterval} from a long[][][][] obtained from a {@link ai.onnxruntime.OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The long[][][][] tensor data is read from.
	 * @return The RandomAccessibleInterval built from the tensor of type {@link LongType}.
	 */
    private static RandomAccessibleInterval<LongType> buildFromTensorLong(long[][][][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length, 
    			tensor[0][0].length, tensor[0][0][0].length};
		int totalSize = 1;
		for (long i : tensorShape) totalSize *= i;
		long[] flatArr = new long[totalSize];
		int cc = 0;
		for (int l = 0; l < tensor[0][0][0].length; l ++) {
			for (int k = 0; k < tensor[0][0].length; k ++) {
				for (int j = 0; j < tensor[0].length; j ++) {
					for (int i = 0; i < tensor.length; i ++) {
						flatArr[cc ++] = tensor[i][j][k][l];
					}
				}
			}
		}
		RandomAccessibleInterval<LongType> rai = ArrayImgs.longs(flatArr, tensorShape);
		return rai;
    }

    /**
	 * Builds a {@link RandomAccessibleInterval} from a long[][][] obtained from a {@link ai.onnxruntime.OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The long[][][] tensor data is read from.
	 * @return The RandomAccessibleInterval built from the tensor of type {@link LongType}.
	 */
    private static RandomAccessibleInterval<LongType> buildFromTensorLong(long[][][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length, 
    			tensor[0][0].length};
		int totalSize = 1;
		for (long i : tensorShape) totalSize *= i;
		long[] flatArr = new long[totalSize];
		int cc = 0;
		for (int k = 0; k < tensor[0][0].length; k ++) {
			for (int j = 0; j < tensor[0].length; j ++) {
				for (int i = 0; i < tensor.length; i ++) {
					flatArr[cc ++] = tensor[i][j][k];
				}
			}
		}
		RandomAccessibleInterval<LongType> rai = ArrayImgs.longs(flatArr, tensorShape);
		return rai;
    }

    /**
	 * Builds a {@link RandomAccessibleInterval} from a long[][] obtained from a {@link ai.onnxruntime.OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The long[][] tensor data is read from.
	 * @return The RandomAccessibleInterval built from the tensor of type {@link LongType}.
	 */
    private static RandomAccessibleInterval<LongType> buildFromTensorLong(long[][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length};
		int totalSize = 1;
		for (long i : tensorShape) totalSize *= i;
		long[] flatArr = new long[totalSize];
		int cc = 0;
		for (int j = 0; j < tensor[0].length; j ++) {
			for (int i = 0; i < tensor.length; i ++) {
				flatArr[cc ++] = tensor[i][j];
			}
		}
		RandomAccessibleInterval<LongType> rai = ArrayImgs.longs(flatArr, tensorShape);
		return rai;
    }

    /**
	 * Builds a {@link RandomAccessibleInterval} from a long[] obtained from a {@link ai.onnxruntime.OnnxTensor}
	 * 
	 * @param tensor 
	 * 	The long[] tensor data is read from.
	 * @return The RandomAccessibleInterval built from the tensor of type {@link LongType}.
	 */
    private static RandomAccessibleInterval<LongType> buildFromTensorLong(long[] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length};
		int totalSize = 1;
		for (long i : tensorShape) totalSize *= i;
		long[] flatArr = new long[totalSize];
		int cc = 0;
		for (int i = 0; i < tensor.length; i ++) {
			flatArr[cc ++] = tensor[i];
		}
		RandomAccessibleInterval<LongType> rai = ArrayImgs.longs(flatArr, tensorShape);
		return rai;
    }
}
