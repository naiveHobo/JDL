package NumJ;

class IncompatibleMatrixException extends Exception {
	public IncompatibleMatrixException(){
		super("The matrices are not compatible!");
	}
}

public class MatrixOps {
	private float[][] matrix1;
	private float[][] matrix2;
	private float[][] result;
	private int rows;
	private int cols;
	private float[] tempRow;
	private Thread[] threadPool;
	

	public float[][] multiply(float[][] mat1, float[][] mat2) throws IncompatibleMatrixException{
		if(mat1[0].length!=mat2.length)
			throw new IncompatibleMatrixException();

		matrix1 = mat1;
		matrix2 = mat2;

		rows = matrix1.length;
		cols = matrix2[0].length;
		result = new float[rows][cols];

		threadPool = new Thread[rows];
		long start = System.nanoTime();

		for(int i=0; i<rows; i++){
			threadPool[i] = new Thread(new MultiplyThread(i));
			threadPool[i].start();
		}
		
		for(int i=0; i<rows; i++){	
			try{
				threadPool[i].join();
			}catch (InterruptedException e){
				System.out.println(e);
			} 
		}

		long end = System.nanoTime();
		double time = (end-start)/1000000.0;

		System.out.println();
		System.out.println("Matrix multiplication took " + time + " milliseconds.");

		return result;
	}

	public float[][] add(float[][] mat1, float[][] mat2) throws IncompatibleMatrixException{
		if(mat1.length!=mat2.length || mat1[0].length!=mat2[0].length)
			throw new IncompatibleMatrixException();

		matrix1 = mat1;
		matrix2 = mat2;

		rows = matrix1.length;
		cols = matrix1[0].length;
		result = new float[rows][cols];

		threadPool = new Thread[rows];
		long start = System.nanoTime();

		for(int i=0; i<rows; i++){
			threadPool[i] = new Thread(new AddThread(i));
			threadPool[i].start();
		}
		
		for(int i=0; i<rows; i++){	
			try{
				threadPool[i].join();
			}catch (InterruptedException e){
				System.out.println(e);
			} 
		}

		long end = System.nanoTime();
		double time = (end-start)/1000000.0;

		System.out.println();
		System.out.println("Matrix addition took " + time + " milliseconds.");

		return result;
	}

	public float dot(float[][] mat1, float[][] mat2) throws IncompatibleMatrixException{
		if(mat1.length!=mat2.length || mat1[0].length!=mat2[0].length)
			throw new IncompatibleMatrixException();

		matrix1 = mat1;
		matrix2 = mat2;

		rows = matrix1.length;
		cols = matrix1[0].length;

		float dot = 0;
		tempRow = new float[rows];

		threadPool = new Thread[rows];
		long start = System.nanoTime();

		for(int i=0; i<rows; i++){
			threadPool[i] = new Thread(new DotThread(i));
			threadPool[i].start();
		}

		for(int i=0; i<rows; i++){	
			try{
				threadPool[i].join();
			}catch (InterruptedException e){
				System.out.println(e);
			}
		}

		for(int i=0;i<rows;i++)
			dot += tempRow[i];

		long end = System.nanoTime();
		double time = (end-start)/1000000.0;

		System.out.println("Dot product: " + dot);
		System.out.println("Dot product took " + time + " milliseconds.");

		return dot;
	}

	public float sum(float[][] mat) {
		matrix1 = mat;

		rows = matrix1.length;
		cols = matrix1[0].length;

		float s = 0;
		tempRow = new float[rows];

		threadPool = new Thread[rows];
		long start = System.nanoTime();

		for(int i=0; i<rows; i++){
			threadPool[i] = new Thread(new SumThread(i));
			threadPool[i].start();
		}
		
		for(int i=0; i<rows; i++){	
			try{
				threadPool[i].join();
			}catch (InterruptedException e){
				System.out.println(e);
			}
		}

		for(int i=0;i<rows;i++)
			s += tempRow[i];

		long end = System.nanoTime();
		double time = (end-start)/1000000.0;

		System.out.println("Sum: " + s);
		System.out.println("Computation took " + time + " milliseconds.");

		return s;
	}

	public float max(float[][] mat) {
		matrix1 = mat;

		rows = matrix1.length;
		cols = matrix1[0].length;

		float m = -Float.MAX_VALUE;
		tempRow = new float[rows];

		threadPool = new Thread[rows];
		long start = System.nanoTime();

		for(int i=0; i<rows; i++){
			threadPool[i] = new Thread(new MaxThread(i, true));
			threadPool[i].start();
		}
		
		for(int i=0; i<rows; i++){	
			try{
				threadPool[i].join();
			}catch (InterruptedException e){
				System.out.println(e);
			}
		}

		for(int i=0;i<rows;i++){
			if(tempRow[i]>m)
				m = tempRow[i];
		}

		long end = System.nanoTime();
		double time = (end-start)/1000000.0;

		System.out.println("Max: " + m);
		System.out.println("Computation took " + time + " milliseconds.");

		return m;
	}

	public int[] argmax(float[][] mat) {
		matrix1 = mat;

		rows = matrix1.length;
		cols = matrix1[0].length;

		float m = -Float.MAX_VALUE;
		int[] pos = new int[2];
		tempRow = new float[rows];

		threadPool = new Thread[rows];
		long start = System.nanoTime();

		for(int i=0; i<rows; i++){
			threadPool[i] = new Thread(new MaxThread(i, false));
			threadPool[i].start();
		}
		
		for(int i=0; i<rows; i++){	
			try{
				threadPool[i].join();
			}catch (InterruptedException e){
				System.out.println(e);
			}
		}

		for(int i=0;i<rows;i++){
			if(matrix1[i][(int)tempRow[i]]>m){
				m = matrix1[i][(int)tempRow[i]];
				pos[0] = i;
				pos[1] = (int)tempRow[i];
			}
		}

		long end = System.nanoTime();
		double time = (end-start)/1000000.0;

		System.out.println("(" + pos[0] + ", " + pos[1] + ")");
		System.out.println("Computation took " + time + " milliseconds.");

		return pos;
	}

	public float[][] exp(float[][] mat) {
		matrix1 = mat;

		rows = matrix1.length;
		cols = matrix1[0].length;

		threadPool = new Thread[rows];
		long start = System.nanoTime();

		for(int i=0; i<rows; i++){
			threadPool[i] = new Thread(new ExpThread(i));
			threadPool[i].start();
		}

		for(int i=0; i<rows; i++){
			try{
				threadPool[i].join();
			}catch(InterruptedException e){
				System.out.println(e);
			}
		}

		long end = System.nanoTime();
		double time = (end-start)/1000000.0;

		return result;
	}

	public float[][] identity(int x, int y){
		float[][] id = new float[x][y];
		for(int i=0; i<x; i++) {
			for(int j=0; j<y; j++)
				id[i][j] = 1.0f;
		}
		return id;
	}

	private class MultiplyThread implements Runnable {
		int index;

		MultiplyThread(int index){
			this.index = index;
		}

		public void run(){
			for(int i=0; i<matrix2[0].length; i++){
				for(int j=0; j<matrix1[0].length; j++){
					result[index][i] += matrix1[index][j] * matrix2[j][i];
				}
			}
		}
	}

	private class AddThread implements Runnable {
		int index;

		AddThread(int index){
			this.index = index;
		}

		public void run(){
			for(int i=0; i<cols; i++){
				result[index][i] = matrix1[index][i] + matrix2[index][i];
			}
		}
	}

	private class SumThread implements Runnable {
		int index;
		float sum;

		SumThread(int index){
			this.index = index;
			sum = 0;
		}

		public void run(){
			for(int i=0; i<cols; i++){
				sum += matrix1[index][i];
			}
			tempRow[index] = sum;
		}
	}

	private class MaxThread implements Runnable {
		int index;
		float max;
		boolean arg;
		int pos;

		MaxThread(int index, boolean arg){
			this.index = index;
			max = Integer.MIN_VALUE;
			this.arg = arg;
		}

		public void run(){
			for(int i=0; i<cols; i++){
				if(matrix1[index][i]>max){
					max = matrix1[index][i];
					pos = i;
				}
			}
			if(arg)
				tempRow[index] = max;
			else
				tempRow[index] = (float)pos;
		}
	}

	private class DotThread implements Runnable {
		int index;
		float sum;

		DotThread(int index){
			this.index = index;
			sum = 0;
		}

		public void run(){
			for(int i=0; i<cols; i++)
				sum += matrix1[index][i] * matrix2[index][i];
			tempRow[index] = sum;
		}
	}

	private class ExpThread implements Runnable {
		int index;

		ExpThread(int index){
			this.index = index;
		}

		public void run(){
			for(int i=0; i<cols; i++){
				result[index][i] = (float) Math.exp(matrix1[index][i]);
			}
		}
	}
}