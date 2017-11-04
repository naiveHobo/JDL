package NumJ;

class IncompatibleMatrixException extends Exception {
	public IncompatibleMatrixException(){
		super("The matrices are not compatible!");
	}
}

public class NumJ {
	private static float[][] matrix1;
	private static float[][] matrix2;
	private static float[][] result;
	private static int rows;
	private static int cols;
	private static float[] tempRow;
	private static Thread[] threadPool;
	

	public static float[][] dot(float[][] mat1, float[][] mat2) throws IncompatibleMatrixException{
		if(mat1[0].length!=mat2.length)
			throw new IncompatibleMatrixException();

		matrix1 = mat1;
		matrix2 = mat2;

		rows = matrix1.length;
		cols = matrix2[0].length;
		result = new float[rows][cols];

		threadPool = new Thread[rows];
		// long start = System.nanoTime();

		for(int i=0; i<rows; i++){
			threadPool[i] = new Thread(new MultiplyThread(i));
			threadPool[i].start();
		}
		
		for(int i=0; i<rows; i++){	
			try{
				threadPool[i].join();
			}catch (InterruptedException e){
				// System.out.println(e);
			} 
		}

		// long end = System.nanoTime();
		// double time = (end-start)/1000000.0;

		// System.out.println("Computation took " + time + " milliseconds.");

		return result;
	}



	// Matrix-Matrix operations

	public static float[][] add(float[][] mat1, float[][] mat2) throws IncompatibleMatrixException{
		if(mat1.length==1 && mat1[0].length==mat2[0].length){
			matrix2 = mat2;
			matrix1 = new float[mat2.length][mat2[0].length];
			for(int i=0;i<mat2.length;i++)
				matrix1[i] = mat1[0];
		}
		else if(mat2.length==1 && mat1[0].length==mat2[0].length){
			matrix1 = mat1;
			matrix2 = new float[mat1.length][mat1[0].length];
			for(int i=0;i<mat1.length;i++)
				matrix2[i] = mat2[0];
		}
		else if(mat1[0].length==1 && mat1.length==mat2.length){
			matrix2 = mat2;
			matrix1 = new float[mat2.length][mat2[0].length];
			for(int i=0;i<mat2.length;i++){
				for(int j=0;j<mat2[0].length;j++)
					matrix1[i][j] = mat1[i][0];
			}
		}
		else if(mat2[0].length==1 && mat1.length==mat2.length){
			matrix1 = mat1;
			matrix2 = new float[mat1.length][mat1[0].length];
			for(int i=0;i<mat1.length;i++){
				for(int j=0;j<mat1[0].length;j++)
					matrix2[i][j] = mat2[i][0];
			}
		}
		else if(mat1.length!=mat2.length || mat1[0].length!=mat2[0].length)
			throw new IncompatibleMatrixException();
		else{
			matrix1 = mat1;
			matrix2 = mat2;
		}

		rows = matrix1.length;
		cols = matrix1[0].length;
		result = new float[rows][cols];

		threadPool = new Thread[rows];
		// long start = System.nanoTime();

		for(int i=0; i<rows; i++){
			threadPool[i] = new Thread(new MatWithMatThread(i, 2));
			threadPool[i].start();
		}
		
		for(int i=0; i<rows; i++){	
			try{
				threadPool[i].join();
			}catch (InterruptedException e){
				// System.out.println(e);
			} 
		}

		// long end = System.nanoTime();
		// double time = (end-start)/1000000.0;

		// System.out.println("Computation took " + time + " milliseconds.");

		return result;
	}

	public static float[][] mul(float[][] mat1, float[][] mat2) throws IncompatibleMatrixException{
		if(mat1.length==1 && mat1[0].length==mat2[0].length){
			matrix2 = mat2;
			matrix1 = new float[mat2.length][mat2[0].length];
			for(int i=0;i<mat2.length;i++)
				matrix1[i] = mat1[0];
		}
		else if(mat2.length==1 && mat1[0].length==mat2[0].length){
			matrix1 = mat1;
			matrix2 = new float[mat1.length][mat1[0].length];
			for(int i=0;i<mat1.length;i++)
				matrix2[i] = mat2[0];
		}
		else if(mat1[0].length==1 && mat1.length==mat2.length){
			matrix2 = mat2;
			matrix1 = new float[mat2.length][mat2[0].length];
			for(int i=0;i<mat2.length;i++){
				for(int j=0;j<mat2[0].length;j++)
					matrix1[i][j] = mat1[i][0];
			}
		}
		else if(mat2[0].length==1 && mat1.length==mat2.length){
			matrix1 = mat1;
			matrix2 = new float[mat1.length][mat1[0].length];
			for(int i=0;i<mat1.length;i++){
				for(int j=0;j<mat1[0].length;j++)
					matrix2[i][j] = mat2[i][0];
			}
		}
		else if(mat1.length!=mat2.length || mat1[0].length!=mat2[0].length)
			throw new IncompatibleMatrixException();
		else{
			matrix1 = mat1;
			matrix2 = mat2;
		}

		rows = matrix1.length;
		cols = matrix1[0].length;
		result = new float[rows][cols];

		threadPool = new Thread[rows];
		// long start = System.nanoTime();

		for(int i=0; i<rows; i++){
			threadPool[i] = new Thread(new MatWithMatThread(i, 0));
			threadPool[i].start();
		}
		
		for(int i=0; i<rows; i++){	
			try{
				threadPool[i].join();
			}catch (InterruptedException e){
				// System.out.println(e);
			} 
		}

		// long end = System.nanoTime();
		// double time = (end-start)/1000000.0;

		// System.out.println("Computation took " + time + " milliseconds.");

		return result;
	}

	public static float[][] div(float[][] mat1, float[][] mat2) throws IncompatibleMatrixException{
		if(mat1.length==1 && mat1[0].length==mat2[0].length){
			matrix2 = mat2;
			matrix1 = new float[mat2.length][mat2[0].length];
			for(int i=0;i<mat2.length;i++)
				matrix1[i] = mat1[0];
		}
		else if(mat2.length==1 && mat1[0].length==mat2[0].length){
			matrix1 = mat1;
			matrix2 = new float[mat1.length][mat1[0].length];
			for(int i=0;i<mat1.length;i++)
				matrix2[i] = mat2[0];
		}
		else if(mat1[0].length==1 && mat1.length==mat2.length){
			matrix2 = mat2;
			matrix1 = new float[mat2.length][mat2[0].length];
			for(int i=0;i<mat2.length;i++){
				for(int j=0;j<mat2[0].length;j++)
					matrix1[i][j] = mat1[i][0];
			}
		}
		else if(mat2[0].length==1 && mat1.length==mat2.length){
			matrix1 = mat1;
			matrix2 = new float[mat1.length][mat1[0].length];
			for(int i=0;i<mat1.length;i++){
				for(int j=0;j<mat1[0].length;j++)
					matrix2[i][j] = mat2[i][0];
			}
		}
		else if(mat1.length!=mat2.length || mat1[0].length!=mat2[0].length)
			throw new IncompatibleMatrixException();
		else{
			matrix1 = mat1;
			matrix2 = mat2;
		}

		rows = matrix1.length;
		cols = matrix1[0].length;
		result = new float[rows][cols];

		threadPool = new Thread[rows];
		// long start = System.nanoTime();

		for(int i=0; i<rows; i++){
			threadPool[i] = new Thread(new MatWithMatThread(i, 1));
			threadPool[i].start();
		}
		
		for(int i=0; i<rows; i++){	
			try{
				threadPool[i].join();
			}catch (InterruptedException e){
				// System.out.println(e);
			} 
		}

		// long end = System.nanoTime();
		// double time = (end-start)/1000000.0;

		// System.out.println("Computation took " + time + " milliseconds.");

		return result;
	}

	public static float[][] sub(float[][] mat1, float[][] mat2) throws IncompatibleMatrixException{
		if(mat1.length==1 && mat1[0].length==mat2[0].length){
			matrix2 = mat2;
			matrix1 = new float[mat2.length][mat2[0].length];
			for(int i=0;i<mat2.length;i++)
				matrix1[i] = mat1[0];
		}
		else if(mat2.length==1 && mat1[0].length==mat2[0].length){
			matrix1 = mat1;
			matrix2 = new float[mat1.length][mat1[0].length];
			for(int i=0;i<mat1.length;i++)
				matrix2[i] = mat2[0];
		}
		else if(mat1[0].length==1 && mat1.length==mat2.length){
			matrix2 = mat2;
			matrix1 = new float[mat2.length][mat2[0].length];
			for(int i=0;i<mat2.length;i++){
				for(int j=0;j<mat2[0].length;j++)
					matrix1[i][j] = mat1[i][0];
			}
		}
		else if(mat2[0].length==1 && mat1.length==mat2.length){
			matrix1 = mat1;
			matrix2 = new float[mat1.length][mat1[0].length];
			for(int i=0;i<mat1.length;i++){
				for(int j=0;j<mat1[0].length;j++)
					matrix2[i][j] = mat2[i][0];
			}
		}
		else if(mat1.length!=mat2.length || mat1[0].length!=mat2[0].length)
			throw new IncompatibleMatrixException();
		else{
			matrix1 = mat1;
			matrix2 = mat2;
		}

		rows = matrix1.length;
		cols = matrix1[0].length;
		result = new float[rows][cols];

		threadPool = new Thread[rows];
		// long start = System.nanoTime();

		for(int i=0; i<rows; i++){
			threadPool[i] = new Thread(new MatWithMatThread(i, 3));
			threadPool[i].start();
		}
		
		for(int i=0; i<rows; i++){	
			try{
				threadPool[i].join();
			}catch (InterruptedException e){
				// System.out.println(e);
			} 
		}

		// long end = System.nanoTime();
		// double time = (end-start)/1000000.0;

		// System.out.println("Computation took " + time + " milliseconds.");

		return result;
	}

	public static float dotProduct(float[][] mat1, float[][] mat2) throws IncompatibleMatrixException{
		if(mat1.length==1 && mat1[0].length==mat2[0].length){
			matrix2 = mat2;
			matrix1 = new float[mat2.length][mat2[0].length];
			for(int i=0;i<mat2.length;i++)
				matrix1[i] = mat1[0];
		}
		else if(mat2.length==1 && mat1[0].length==mat2[0].length){
			matrix1 = mat1;
			matrix2 = new float[mat1.length][mat1[0].length];
			for(int i=0;i<mat1.length;i++)
				matrix2[i] = mat2[0];
		}
		else if(mat1[0].length==1 && mat1.length==mat2.length){
			matrix2 = mat2;
			matrix1 = new float[mat2.length][mat2[0].length];
			for(int i=0;i<mat2.length;i++){
				for(int j=0;j<mat2[0].length;j++)
					matrix1[i][j] = mat1[i][0];
			}
		}
		else if(mat2[0].length==1 && mat1.length==mat2.length){
			matrix1 = mat1;
			matrix2 = new float[mat1.length][mat1[0].length];
			for(int i=0;i<mat1.length;i++){
				for(int j=0;j<mat1[0].length;j++)
					matrix2[i][j] = mat2[i][0];
			}
		}
		else if(mat1.length!=mat2.length || mat1[0].length!=mat2[0].length)
			throw new IncompatibleMatrixException();
		else{
			matrix1 = mat1;
			matrix2 = mat2;
		}

		rows = matrix1.length;
		cols = matrix1[0].length;

		float dot = 0;
		tempRow = new float[rows];

		threadPool = new Thread[rows];
		// long start = System.nanoTime();

		for(int i=0; i<rows; i++){
			threadPool[i] = new Thread(new ScalarThread(i, 1));
			threadPool[i].start();
		}

		for(int i=0; i<rows; i++){	
			try{
				threadPool[i].join();
			}catch (InterruptedException e){
				// System.out.println(e);
			}
		}

		for(int i=0;i<rows;i++)
			dot += tempRow[i];

		// long end = System.nanoTime();
		// double time = (end-start)/1000000.0;

		// System.out.println("Dot product: " + dot);
		// System.out.println("Computation took " + time + " milliseconds.");

		return dot;
	}


	// Matrix-Scalar operations

	public static float[][] add(float[][] mat, float sclr) {
		matrix1 = mat;

		rows = matrix1.length;
		cols = matrix1[0].length;

		result = new float[rows][cols];

		threadPool = new Thread[rows];
		// long start = System.nanoTime();

		for(int i=0; i<rows; i++){
			threadPool[i] = new Thread(new MatWithScalarThread(i, 3, sclr));
			threadPool[i].start();
		}

		for(int i=0; i<rows; i++){
			try{
				threadPool[i].join();
			}catch(InterruptedException e){
				// System.out.println(e);
			}
		}

		// long end = System.nanoTime();
		// double time = (end-start)/1000000.0;

		// System.out.println("Computation took " + time + " milliseconds.");

		return result;
	}

	public static float[][] mul(float[][] mat, float sclr) {
		matrix1 = mat;

		rows = matrix1.length;
		cols = matrix1[0].length;

		result = new float[rows][cols];

		threadPool = new Thread[rows];
		// long start = System.nanoTime();

		for(int i=0; i<rows; i++){
			threadPool[i] = new Thread(new MatWithScalarThread(i, 1, sclr));
			threadPool[i].start();
		}

		for(int i=0; i<rows; i++){
			try{
				threadPool[i].join();
			}catch(InterruptedException e){
				// System.out.println(e);
			}
		}

		// long end = System.nanoTime();
		// double time = (end-start)/1000000.0;

		// System.out.println("Computation took " + time + " milliseconds.");

		return result;
	}

	public static float[][] div(float[][] mat, float sclr) {
		matrix1 = mat;

		rows = matrix1.length;
		cols = matrix1[0].length;

		result = new float[rows][cols];

		threadPool = new Thread[rows];
		// long start = System.nanoTime();

		for(int i=0; i<rows; i++){
			threadPool[i] = new Thread(new MatWithScalarThread(i, 2, sclr));
			threadPool[i].start();
		}

		for(int i=0; i<rows; i++){
			try{
				threadPool[i].join();
			}catch(InterruptedException e){
				// System.out.println(e);
			}
		}

		// long end = System.nanoTime();
		// double time = (end-start)/1000000.0;

		// System.out.println("Computation took " + time + " milliseconds.");

		return result;
	}

	public static float[][] sub(float[][] mat, float sclr) {
		matrix1 = mat;

		rows = matrix1.length;
		cols = matrix1[0].length;
		
		result = new float[rows][cols];

		threadPool = new Thread[rows];
		// long start = System.nanoTime();

		for(int i=0; i<rows; i++){
			threadPool[i] = new Thread(new MatWithScalarThread(i, 4, sclr));
			threadPool[i].start();
		}

		for(int i=0; i<rows; i++){
			try{
				threadPool[i].join();
			}catch(InterruptedException e){
				// System.out.println(e);
			}
		}

		// long end = System.nanoTime();
		// double time = (end-start)/1000000.0;

		// System.out.println("Computation took " + time + " milliseconds.");

		return result;
	}

	

	// Scalar output operations

	public static float sum(float[][] mat) {
		matrix1 = mat;

		rows = matrix1.length;
		cols = matrix1[0].length;

		float s = 0;
		tempRow = new float[rows];

		threadPool = new Thread[rows];
		// long start = System.nanoTime();

		for(int i=0; i<rows; i++){
			threadPool[i] = new Thread(new ScalarThread(i, 0));
			threadPool[i].start();
		}
		
		for(int i=0; i<rows; i++){	
			try{
				threadPool[i].join();
			}catch (InterruptedException e){
				// System.out.println(e);
			}
		}

		for(int i=0;i<rows;i++)
			s += tempRow[i];

		// long end = System.nanoTime();
		// double time = (end-start)/1000000.0;

		// System.out.println("Sum: " + s);
		// System.out.println("Computation took " + time + " milliseconds.");

		return s;
	}

	public static float[][] sum(float[][] mat, int axis) {
		if(axis==1)
			matrix1 = mat;
		else if(axis==0)
			matrix1 = transpose(mat);

		rows = matrix1.length;
		cols = matrix1[0].length;

		tempRow = new float[rows];

		threadPool = new Thread[rows];
		// long start = System.nanoTime();

		for(int i=0; i<rows; i++){
			threadPool[i] = new Thread(new ScalarThread(i, 0));
			threadPool[i].start();
		}
		
		for(int i=0; i<rows; i++){	
			try{
				threadPool[i].join();
			}catch (InterruptedException e){
				// System.out.println(e);
			}
		}

		result = new float[1][tempRow.length];
		result[0] = tempRow;	

		// long end = System.nanoTime();
		// double time = (end-start)/1000000.0;

		// System.out.println("Sum: " + s);
		// System.out.println("Computation took " + time + " milliseconds.");

		return result;
	}

	public static float avg(float[][] mat) {
		float av = sum(mat);
		av /= (mat.length * mat[0].length);
		return av;
	}

	public static float[][] avg(float[][] mat, int axis) {
		float[][] temp = sum(mat, axis);
		int n = axis==0 ? mat.length : mat[0].length;
		for(int i=0;i<temp.length;i++)
			temp[0][i] /= n;
		return temp;
	}

	public static float max(float[][] mat) {
		matrix1 = mat;

		rows = matrix1.length;
		cols = matrix1[0].length;

		float m = -Float.MAX_VALUE;
		tempRow = new float[rows];

		threadPool = new Thread[rows];
		// long start = System.nanoTime();

		for(int i=0; i<rows; i++){
			threadPool[i] = new Thread(new ScalarThread(i, 2));
			threadPool[i].start();
		}
		
		for(int i=0; i<rows; i++){	
			try{
				threadPool[i].join();
			}catch (InterruptedException e){
				// System.out.println(e);
			}
		}

		for(int i=0;i<rows;i++){
			if(tempRow[i]>m)
				m = tempRow[i];
		}

		// long end = System.nanoTime();
		// double time = (end-start)/1000000.0;

		// System.out.println("Max: " + m);
		// System.out.println("Computation took " + time + " milliseconds.");

		return m;
	}

	public static float[][] max(float[][] mat, int axis) {
		if(axis==1)
			matrix1 = mat;
		else if(axis==0)
			matrix1 = transpose(mat);

		rows = matrix1.length;
		cols = matrix1[0].length;

		tempRow = new float[rows];

		threadPool = new Thread[rows];
		// long start = System.nanoTime();

		for(int i=0; i<rows; i++){
			threadPool[i] = new Thread(new ScalarThread(i, 2));
			threadPool[i].start();
		}
		
		for(int i=0; i<rows; i++){	
			try{
				threadPool[i].join();
			}catch (InterruptedException e){
				// System.out.println(e);
			}
		}

		// long end = System.nanoTime();
		// double time = (end-start)/1000000.0;

		// System.out.println("Max: " + m);
		// System.out.println("Computation took " + time + " milliseconds.");

		float[][] temp = new float[1][rows];
		temp[0] = tempRow;

		return temp;
	}

	public static int[] argmax(float[][] mat) {
		matrix1 = mat;

		rows = matrix1.length;
		cols = matrix1[0].length;

		float m = -Float.MAX_VALUE;
		int[] pos = new int[2];
		tempRow = new float[rows];

		threadPool = new Thread[rows];
		// long start = System.nanoTime();

		for(int i=0; i<rows; i++){
			threadPool[i] = new Thread(new ScalarThread(i, 3));
			threadPool[i].start();
		}
		
		for(int i=0; i<rows; i++){	
			try{
				threadPool[i].join();
			}catch (InterruptedException e){
				// System.out.println(e);
			}
		}

		for(int i=0;i<rows;i++){
			if(matrix1[i][(int)tempRow[i]]>m){
				m = matrix1[i][(int)tempRow[i]];
				pos[0] = i;
				pos[1] = (int)tempRow[i];
			}
		}

		// long end = System.nanoTime();
		// double time = (end-start)/1000000.0;

		// System.out.println("(" + pos[0] + ", " + pos[1] + ")");
		// System.out.println("Computation took " + time + " milliseconds.");

		return pos;
	}

	public static int[] argmax(float[][] mat, int axis) {
		if(axis==1)
			matrix1 = mat;
		else if(axis==0)
			matrix1 = transpose(mat);

		rows = matrix1.length;
		cols = matrix1[0].length;

		int[] pos = new int[rows];
		tempRow = new float[rows];

		threadPool = new Thread[rows];
		// long start = System.nanoTime();

		for(int i=0; i<rows; i++){
			threadPool[i] = new Thread(new ScalarThread(i, 3));
			threadPool[i].start();
		}
		
		for(int i=0; i<rows; i++){	
			try{
				threadPool[i].join();
			}catch (InterruptedException e){
				// System.out.println(e);
			}
		}

		for(int i=0;i<rows;i++)
			pos[i] = (int)tempRow[i];

		// long end = System.nanoTime();
		// double time = (end-start)/1000000.0;

		// System.out.println("(" + pos[0] + ", " + pos[1] + ")");
		// System.out.println("Computation took " + time + " milliseconds.");

		return pos;
	}


	// Single matrix operations

	public static float[][] exp(float[][] mat) {
		matrix1 = mat;

		rows = matrix1.length;
		cols = matrix1[0].length;

		result = new float[rows][cols];

		threadPool = new Thread[rows];
		// long start = System.nanoTime();

		for(int i=0; i<rows; i++){
			threadPool[i] = new Thread(new MatWithScalarThread(i, 0, 0));
			threadPool[i].start();
		}

		for(int i=0; i<rows; i++){
			try{
				threadPool[i].join();
			}catch(InterruptedException e){
				// System.out.println(e);
			}
		}

		// long end = System.nanoTime();
		// double time = (end-start)/1000000.0;

		// System.out.println("Computation took " + time + " milliseconds.");

		return result;
	}

	public static float[][] transpose(float[][] mat) {
		matrix1 = mat;

		rows = matrix1.length;
		cols = matrix1[0].length;

		result = new float[cols][rows];

		threadPool = new Thread[rows];
		// long start = System.nanoTime();

		for(int i=0; i<rows; i++){
			threadPool[i] = new Thread(new MatWithScalarThread(i, 5, 0));
			threadPool[i].start();
		}

		for(int i=0; i<rows; i++){
			try{
				threadPool[i].join();
			}catch(InterruptedException e){
				// System.out.println(e);
			}
		}

		// long end = System.nanoTime();
		// double time = (end-start)/1000000.0;

		// System.out.println("Computation took " + time + " milliseconds.");

		return result;
	}

	public static float[][] neg(float[][] mat) {
		matrix1 = mat;

		rows = matrix1.length;
		cols = matrix1[0].length;

		result = new float[rows][cols];

		threadPool = new Thread[rows];
		// long start = System.nanoTime();

		for(int i=0; i<rows; i++){
			threadPool[i] = new Thread(new MatWithScalarThread(i, 6, 0));
			threadPool[i].start();
		}

		for(int i=0; i<rows; i++){
			try{
				threadPool[i].join();
			}catch(InterruptedException e){
				// System.out.println(e);
			}
		}

		// long end = System.nanoTime();
		// double time = (end-start)/1000000.0;

		// System.out.println("Computation took " + time + " milliseconds.");

		return result;
	}

	public static float[][] reciprocal(float[][] mat) {
		matrix1 = mat;

		rows = matrix1.length;
		cols = matrix1[0].length;

		result = new float[rows][cols];

		threadPool = new Thread[rows];
		// long start = System.nanoTime();

		for(int i=0; i<rows; i++){
			threadPool[i] = new Thread(new MatWithScalarThread(i, 7, 0));
			threadPool[i].start();
		}

		for(int i=0; i<rows; i++){
			try{
				threadPool[i].join();
			}catch(InterruptedException e){
				// System.out.println(e);
			}
		}

		// long end = System.nanoTime();
		// double time = (end-start)/1000000.0;

		// System.out.println("Computation took " + time + " milliseconds.");

		return result;
	}

	public static float[][] identity(int size){
		float[][] id = new float[size][size];
		for(int i=0; i<size; i++) {
			id[i][i] = 1.0f;
		}
		return id;
	}

	public static float[][] ones(int x, int y){
		rows = x;
		cols = y;

		result = new float[rows][cols];

		threadPool = new Thread[rows];
		// long start = System.nanoTime();

		for(int i=0; i<rows; i++){
			threadPool[i] = new Thread(new MatWithScalarThread(i, 8, 0));
			threadPool[i].start();
		}

		for(int i=0; i<rows; i++){
			try{
				threadPool[i].join();
			}catch(InterruptedException e){
				// System.out.println(e);
			}
		}

		// long end = System.nanoTime();
		// double time = (end-start)/1000000.0;

		// System.out.println("Computation took " + time + " milliseconds.");

		return result;
	}

	public static float[][] ones(int size){
		float[][] temp = new float[1][size];
		for(int i=0;i<size;i++)
			temp[0][i] = 1.0f;
		return temp;
	}

	public static float[][] zeros(int x, int y){
		rows = x;
		cols = y;

		result = new float[rows][cols];

		return result;
	}

	public static float[][] zeros(int size){
		return (new float[1][size]);
	}

	public static float[][] rand(int x, int y){
		rows = x;
		cols = y;

		result = new float[rows][cols];

		threadPool = new Thread[rows];
		// long start = System.nanoTime();

		for(int i=0; i<rows; i++){
			threadPool[i] = new Thread(new MatWithScalarThread(i, 9, 0));
			threadPool[i].start();
		}

		for(int i=0; i<rows; i++){
			try{
				threadPool[i].join();
			}catch(InterruptedException e){
				// System.out.println(e);
			}
		}

		// long end = System.nanoTime();
		// double time = (end-start)/1000000.0;

		// System.out.println("Computation took " + time + " milliseconds.");

		return result;
	}

	public static float[][] rand(int size){
		float[][] temp = new float[1][size];
		for(int i=0;i<size;i++)
			temp[0][i] = ((float) Math.random() * 2.0f) - 1.0f;
		return temp;
	}

	public static float[][] flatten(float[][] mat) {
		cols = mat.length * mat[0].length;

		float[][] temp = new float[1][cols];
		int cnt = 0;
		for(int i=0;i<mat.length;i++){
			for(int j=0;j<mat[0].length;j++)
				temp[0][cnt++] = mat[i][j];
		}

		return temp;
	}

	public static float[][] reshape(float[][] mat, int[] newshape) {
		if(newshape.length!=2)
			return mat;
		int[] curshape = shape(mat);
		if(curshape[0]*curshape[1]!=newshape[0]*newshape[1])
			return mat;

		float[][] out = new float[newshape[0]][newshape[1]];
		float[][] temp = flatten(mat);
		int cnt = 0;
		for(int i=0;i<newshape[0];i++){
			for(int j=0;j<newshape[1];j++)
				out[i][j] = temp[0][cnt++];
		}
		return out;
	}

	public static float[][] reshape(float[][] mat, int newshapex, int newshapey) {
		int[] newshape = {newshapex, newshapey};
		return reshape(mat, newshape);
	}

	public static int[] shape(float[][] mat) {
		int[] s = new int[2];
		s[0] = mat.length;
		s[1] = mat[0].length;
		return s;
	}

	public static float[][] copy(float[][] mat) {
		float[][] temp = new float[mat.length][mat[0].length];
		for(int i=0;i<mat.length;i++){
			for(int j=0;j<mat[0].length;j++)
				temp[i][j] = mat[i][j];
		}
		return temp;
	}

	public static void display(float[][] mat) {
		for(int i=0;i<mat.length;i++){
			for(int j=0;j<mat[0].length;j++)
				System.out.print(mat[i][j] + " ");
			System.out.println();
		}
		System.out.println();
	}

	private static class MultiplyThread implements Runnable {
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

	private static class ScalarThread implements Runnable {
		int index;
		float temp;
		int type;

		ScalarThread(int index, int type){
			this.index = index;
			if(type==0 || type==1)
				temp = 0;
			else
				temp = -Float.MAX_VALUE;
			this.type = type;
		}

		public void run(){
			int pos = 0;
			for(int i=0; i<cols; i++){
				if(type==0)
					temp += matrix1[index][i];
				else if(type==1)
					temp += matrix1[index][i] * matrix2[index][i];
				else{
					if(matrix1[index][i]>temp){
						temp = matrix1[index][i];
						pos = i;
					}
				}
			}
			if(type==3)
				tempRow[index] = (float) pos;
			else
				tempRow[index] = temp;
		}
	}

	private static class MatWithScalarThread implements Runnable {
		int index;
		int type;
		float scalar;

		MatWithScalarThread(int index, int type, float scalar){
			this.index = index;
			this.type = type;
			this.scalar = scalar;
		}

		public void run(){
			for(int i=0; i<cols; i++){
				if(type==0)
					result[index][i] = (float) Math.exp(matrix1[index][i]);
				else if(type==1)
					result[index][i] = matrix1[index][i] * scalar;
				else if(type==2)
					result[index][i] = matrix1[index][i] / scalar;
				else if(type==3)
					result[index][i] = matrix1[index][i] + scalar;
				else if(type==4)
					result[index][i] = matrix1[index][i] - scalar;
				else if(type==5)
					result[i][index] = matrix1[index][i];
				else if(type==6)
					result[index][i] = -matrix1[index][i];
				else if(type==7)
					result[index][i] = 1/matrix1[index][i];
				else if(type==8)
					result[index][i] = 1.0f;
				else if(type==9)
					result[index][i] = ((float) Math.random() * 2.0f) - 1.0f;
			}
		}
	}

	private static class MatWithMatThread implements Runnable {
		int index;
		int type;

		MatWithMatThread(int index, int type){
			this.index = index;
			this.type = type;
		}

		public void run(){
			for(int i=0; i<cols; i++){
				if(type==0)
					result[index][i] = matrix1[index][i] * matrix2[index][i];
				else if(type==1)
					result[index][i] = matrix1[index][i] / matrix2[index][i];
				else if(type==2)
					result[index][i] = matrix1[index][i] + matrix2[index][i];
				else if(type==3)
					result[index][i] = matrix1[index][i] - matrix2[index][i];
			}
		}
	}
}