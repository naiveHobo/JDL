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

	MatrixOps(float[][] mat1, float[][] mat2) {
		matrix1 = mat1;
		matrix2 = mat2;
	}

	MatrixOps(float[][] mat1) {
		matrix1 = mat1;
		matrix2 = mat1;
		rows = matrix1.length;
		cols = matrix1[0].length;
		result = new float[rows][cols];
	}

	public float[][] multiply() throws IncompatibleMatrixException{
        if(matrix1[0].length!=matrix2.length)
			throw new IncompatibleMatrixException();

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
       System.out.println("Multiplication took " + time + " milliseconds.");

       return result;
    }

    public float[][] add() throws IncompatibleMatrixException{
        if(matrix1.length!=matrix2.length || matrix1[0].length!=matrix2[0].length)
    		throw new IncompatibleMatrixException();

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
       System.out.println("Addition took " + time + " milliseconds.");

       return result;
    }

    public float dotProduct() throws IncompatibleMatrixException{
    	if(matrix1.length!=matrix2.length || matrix1[0].length!=matrix2[0].length)
    		throw new IncompatibleMatrixException();

    	float dot = 0;
    	tempRow = new float[rows];

        threadPool = new Thread[rows];
        long start = System.nanoTime();

        for(int i=0; i<rows; i++){
 			threadPool[i] = new Thread(new DotProductThread(i));
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

    public float sum() throws IncompatibleMatrixException{
    	if(matrix1.length!=matrix2.length || matrix1[0].length!=matrix2[0].length)
    		throw new IncompatibleMatrixException();

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
       System.out.println("Dot product took " + time + " milliseconds.");

       return s;
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

    private class DotProductThread implements Runnable {
        int index;
        float sum;
        
        DotProductThread(int index){
            this.index = index;
            sum = 0;
        }
        
        public void run(){
            for(int i=0; i<cols; i++){
				sum += matrix1[index][i] * matrix2[index][i];
            }
            tempRow[index] = sum;
        }
    }   
}