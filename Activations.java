import NumJ.*; 

public class Activations {
	static MatrixOps m;

	static void display(float[][] mat1) {
		for(int i=0;i<mat1.length;i++) {
			for(int j=0;j<mat1[i].length;j++) {
				System.out.print(mat1[i][j]+" ");
			}
			System.out.println();
		}
	}

	static float[][] sigmoid(float[][] mat1) {
		return m.reciprocal(m.add(m.exp(m.neg(mat1)), 1));

	}

	static float[][] tanh(float[][] mat1) {
		return m.sub(m.mul(sigmoid(m.mul(mat1,2)),2),1);
	}

	static float[][] relu(float[][] mat1) {
		float[][] relu = new float[mat1.length][mat1[0].length];
		for(int i=0;i<mat1.length;i++)
			for(int j=0;j<mat1[i].length;j++)
				relu[i][j] = mat1[i][j]>0?mat1[i][j]:0;
		return relu;
	}

	static float[][] softmax(float[][] mat1) {
		float[][] exp = m.exp(mat1);
		return m.div(exp,m.sum(exp));		
	}

	public static void main(String args[]) {
		m = new MatrixOps();
		float[][] arr = {{1,2,3},
						 {4,5,6},
						 {7,8,9}};
		// float[][] sig = sigmoid(arr);
		// display(sig);
		// float[][] tanh = tanh(arr);
		// display(tanh);
		// float[][] relu = relu(m.neg(tanh));
		// display(relu);
		float[][] softmax = softmax(arr);
		display(softmax);
		System.out.println(m.sum(softmax));
		// display(m.exp(arr));
	}
}