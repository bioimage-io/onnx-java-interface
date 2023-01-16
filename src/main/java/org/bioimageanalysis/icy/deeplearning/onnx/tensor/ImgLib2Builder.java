package org.bioimageanalysis.icy.deeplearning.onnx.tensor;


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
 * A {@link Img} builder for TensorFlow {@link Tensor} objects.
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
     * Creates a {@link Img} from a given {@link Tensor} and an array with its dimensions order.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The INDArray built from the tensor.
     * @throws IllegalArgumentException
     *         If the tensor type is not supported.
     */
    @SuppressWarnings("unchecked")
    public static <T extends Type<T>> Img<T> build(Object tensor) throws IllegalArgumentException
    {
        // Create an INDArray of the same type of the tensor
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
     * Builds a {@link Img} from a unsigned byte-typed {@link Tensor}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The INDArray built from the tensor of type {@link DataType#UBYTE}.
     */
    private static Img<ByteType> buildFromTensorByte(byte[] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length};
    	final ImgFactory< ByteType > factory = new CellImgFactory<>( new ByteType(), 5 );
        final Img< ByteType > outputImg = (Img<ByteType>) factory.create(tensorShape);
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
     * Builds a {@link Img} from a unsigned byte-typed {@link Tensor}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The INDArray built from the tensor of type {@link DataType#UBYTE}.
     */
    private static Img<ByteType> buildFromTensorByte(byte[][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length};
    	final ImgFactory< ByteType > factory = new CellImgFactory<>( new ByteType(), 5 );
        final Img< ByteType > outputImg = (Img<ByteType>) factory.create(tensorShape);
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
     * Builds a {@link Img} from a unsigned byte-typed {@link Tensor}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The INDArray built from the tensor of type {@link DataType#UBYTE}.
     */
    private static Img<ByteType> buildFromTensorByte(byte[][][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length, tensor[1].length};
    	final ImgFactory< ByteType > factory = new CellImgFactory<>( new ByteType(), 5 );
        final Img< ByteType > outputImg = (Img<ByteType>) factory.create(tensorShape);
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
     * Builds a {@link Img} from a unsigned byte-typed {@link Tensor}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The INDArray built from the tensor of type {@link DataType#UBYTE}.
     */
    private static Img<ByteType> buildFromTensorByte(byte[][][][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length, tensor[1].length, tensor[2].length};
    	final ImgFactory< ByteType > factory = new CellImgFactory<>( new ByteType(), 5 );
        final Img< ByteType > outputImg = (Img<ByteType>) factory.create(tensorShape);
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
     * Builds a {@link Img} from a unsigned byte-typed {@link Tensor}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The INDArray built from the tensor of type {@link DataType#UBYTE}.
     */
    private static Img<ByteType> buildFromTensorByte(byte[][][][][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length, 
    			tensor[1].length, tensor[2].length, tensor[3].length};
    	final ImgFactory< ByteType > factory = new CellImgFactory<>( new ByteType(), 5 );
        final Img< ByteType > outputImg = (Img<ByteType>) factory.create(tensorShape);
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
     * Builds a {@link Img} from a unsigned integer-typed {@link Tensor}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The sequence built from the tensor of type {@link DataType#INT}.
     */
    private static Img<IntType> buildFromTensorInt(int[][][][][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length, 
    			tensor[1].length, tensor[2].length, tensor[3].length};
    	final ImgFactory< IntType > factory = new CellImgFactory<>( new IntType(), 5 );
        final Img< IntType > outputImg = (Img<IntType>) factory.create(tensorShape);
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
     * Builds a {@link Img} from a unsigned integer-typed {@link Tensor}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The sequence built from the tensor of type {@link DataType#INT}.
     */
    private static Img<IntType> buildFromTensorInt(int[][][][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length, 
    			tensor[1].length, tensor[2].length};
    	final ImgFactory< IntType > factory = new CellImgFactory<>( new IntType(), 5 );
        final Img< IntType > outputImg = (Img<IntType>) factory.create(tensorShape);
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
     * Builds a {@link Img} from a unsigned integer-typed {@link Tensor}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The sequence built from the tensor of type {@link DataType#INT}.
     */
    private static Img<IntType> buildFromTensorInt(int[][][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length, 
    			tensor[1].length};
    	final ImgFactory< IntType > factory = new CellImgFactory<>( new IntType(), 5 );
        final Img< IntType > outputImg = (Img<IntType>) factory.create(tensorShape);
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
     * Builds a {@link Img} from a unsigned integer-typed {@link Tensor}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The sequence built from the tensor of type {@link DataType#INT}.
     */
    private static Img<IntType> buildFromTensorInt(int[][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length};
    	final ImgFactory< IntType > factory = new CellImgFactory<>( new IntType(), 5 );
        final Img< IntType > outputImg = (Img<IntType>) factory.create(tensorShape);
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
     * Builds a {@link Img} from a unsigned integer-typed {@link Tensor}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The sequence built from the tensor of type {@link DataType#INT}.
     */
    private static Img<IntType> buildFromTensorInt(int[] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length};
    	final ImgFactory< IntType > factory = new CellImgFactory<>( new IntType(), 5 );
        final Img< IntType > outputImg = (Img<IntType>) factory.create(tensorShape);
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
     * Builds a {@link Img} from a unsigned float-typed {@link Tensor}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The INDArray built from the tensor of type {@link DataType#FLOAT}.
     */
    private static Img<FloatType> buildFromTensorFloat(float[][][][][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length, 
    			tensor[1].length, tensor[2].length, tensor[3].length};
    	final ImgFactory< FloatType > factory = new CellImgFactory<>( new FloatType(), 5 );
        final Img< FloatType > outputImg = (Img<FloatType>) factory.create(tensorShape);
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
     * Builds a {@link Img} from a unsigned float-typed {@link Tensor}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The INDArray built from the tensor of type {@link DataType#FLOAT}.
     */
    private static Img<FloatType> buildFromTensorFloat(float[][][][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length, 
    			tensor[1].length, tensor[2].length};
    	final ImgFactory< FloatType > factory = new CellImgFactory<>( new FloatType(), 5 );
        final Img< FloatType > outputImg = (Img<FloatType>) factory.create(tensorShape);
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
     * Builds a {@link Img} from a unsigned float-typed {@link Tensor}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The INDArray built from the tensor of type {@link DataType#FLOAT}.
     */
    private static Img<FloatType> buildFromTensorFloat(float[][][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length, 
    			tensor[1].length};
    	final ImgFactory< FloatType > factory = new CellImgFactory<>( new FloatType(), 5 );
        final Img< FloatType > outputImg = (Img<FloatType>) factory.create(tensorShape);
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
     * Builds a {@link Img} from a unsigned float-typed {@link Tensor}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The INDArray built from the tensor of type {@link DataType#FLOAT}.
     */
    private static Img<FloatType> buildFromTensorFloat(float[][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length};
    	final ImgFactory< FloatType > factory = new CellImgFactory<>( new FloatType(), 5 );
        final Img< FloatType > outputImg = (Img<FloatType>) factory.create(tensorShape);
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
     * Builds a {@link Img} from a unsigned float-typed {@link Tensor}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The INDArray built from the tensor of type {@link DataType#FLOAT}.
     */
    private static Img<FloatType> buildFromTensorFloat(float[] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length};
    	final ImgFactory< FloatType > factory = new CellImgFactory<>( new FloatType(), 5 );
        final Img< FloatType > outputImg = (Img<FloatType>) factory.create(tensorShape);
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
     * Builds a {@link Img} from a unsigned double-typed {@link Tensor}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The INDArray built from the tensor of type {@link DataType#DOUBLE}.
     */
    private static Img<DoubleType> buildFromTensorDouble(double[][][][][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length, 
    			tensor[1].length, tensor[2].length, tensor[3].length};
    	final ImgFactory< DoubleType > factory = new CellImgFactory<>( new DoubleType(), 5 );
        final Img< DoubleType > outputImg = (Img<DoubleType>) factory.create(tensorShape);
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
     * Builds a {@link Img} from a unsigned double-typed {@link Tensor}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The INDArray built from the tensor of type {@link DataType#DOUBLE}.
     */
    private static Img<DoubleType> buildFromTensorDouble(double[][][][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length, 
    			tensor[1].length, tensor[2].length};
    	final ImgFactory< DoubleType > factory = new CellImgFactory<>( new DoubleType(), 5 );
        final Img< DoubleType > outputImg = (Img<DoubleType>) factory.create(tensorShape);
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
     * Builds a {@link Img} from a unsigned double-typed {@link Tensor}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The INDArray built from the tensor of type {@link DataType#DOUBLE}.
     */
    private static Img<DoubleType> buildFromTensorDouble(double[][][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length, 
    			tensor[1].length};
    	final ImgFactory< DoubleType > factory = new CellImgFactory<>( new DoubleType(), 5 );
        final Img< DoubleType > outputImg = (Img<DoubleType>) factory.create(tensorShape);
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
     * Builds a {@link Img} from a unsigned double-typed {@link Tensor}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The INDArray built from the tensor of type {@link DataType#DOUBLE}.
     */
    private static Img<DoubleType> buildFromTensorDouble(double[][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length};
    	final ImgFactory< DoubleType > factory = new CellImgFactory<>( new DoubleType(), 5 );
        final Img< DoubleType > outputImg = (Img<DoubleType>) factory.create(tensorShape);
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
     * Builds a {@link Img} from a unsigned double-typed {@link Tensor}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The INDArray built from the tensor of type {@link DataType#DOUBLE}.
     */
    private static Img<DoubleType> buildFromTensorDouble(double[] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length};
    	final ImgFactory< DoubleType > factory = new CellImgFactory<>( new DoubleType(), 5 );
        final Img< DoubleType > outputImg = (Img<DoubleType>) factory.create(tensorShape);
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
     * Builds a {@link Img} from a unsigned double-typed {@link Tensor}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The INDArray built from the tensor of type {@link DataType#DOUBLE}.
     */
    private static Img<LongType> buildFromTensorLong(long[][][][][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length, 
    			tensor[1].length, tensor[2].length, tensor[3].length};
    	final ImgFactory< LongType > factory = new CellImgFactory<>( new LongType(), 5 );
        final Img< LongType > outputImg = (Img<LongType>) factory.create(tensorShape);
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
     * Builds a {@link Img} from a unsigned double-typed {@link Tensor}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The INDArray built from the tensor of type {@link DataType#DOUBLE}.
     */
    private static Img<LongType> buildFromTensorLong(long[][][][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length, 
    			tensor[1].length, tensor[2].length};
    	final ImgFactory< LongType > factory = new CellImgFactory<>( new LongType(), 5 );
        final Img< LongType > outputImg = (Img<LongType>) factory.create(tensorShape);
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
     * Builds a {@link Img} from a unsigned double-typed {@link Tensor}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The INDArray built from the tensor of type {@link DataType#DOUBLE}.
     */
    private static Img<LongType> buildFromTensorLong(long[][][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length, 
    			tensor[1].length};
    	final ImgFactory< LongType > factory = new CellImgFactory<>( new LongType(), 5 );
        final Img< LongType > outputImg = (Img<LongType>) factory.create(tensorShape);
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
     * Builds a {@link Img} from a unsigned double-typed {@link Tensor}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The INDArray built from the tensor of type {@link DataType#DOUBLE}.
     */
    private static Img<LongType> buildFromTensorLong(long[][] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length, tensor[0].length};
    	final ImgFactory< LongType > factory = new CellImgFactory<>( new LongType(), 5 );
        final Img< LongType > outputImg = (Img<LongType>) factory.create(tensorShape);
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
     * Builds a {@link Img} from a unsigned double-typed {@link Tensor}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The INDArray built from the tensor of type {@link DataType#DOUBLE}.
     */
    private static Img<LongType> buildFromTensorLong(long[] tensor)
    {
    	long[] tensorShape = new long[] {tensor.length};
    	final ImgFactory< LongType > factory = new CellImgFactory<>( new LongType(), 5 );
        final Img< LongType > outputImg = (Img<LongType>) factory.create(tensorShape);
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
