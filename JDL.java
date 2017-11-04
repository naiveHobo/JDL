import NumJ.*;

interface Layer {
	public float[][] forward(float[][] x);
	public float[][] backward(float[][] d);
	public boolean hasWeights();
	public void updateWeights(float lr, float rs);
	public String getName();
	public String getType();
}

class Activation implements Layer {
	public String name;
	public String type;
	private String function;
	private float[][] output;

	public Activation(String name, String function) {
		this.name = name;
		this.type = "Activation";
		this.function = function;
	}

	public float[][] forward(float[][] x) {
		if(function.equals("relu"))
			output = relu(x);
		else if(function.equals("tanh"))
			output = tanh(x);
		else if(function.equals("sigmoid"))
			output = sigmoid(x);
		else if(function.equals("softmax"))
			output = softmax(x);
		return output;
	}

	public float[][] backward(float[][] d) {
		float[][] gradient = new float[d.length][d[0].length];
		if(function.equals("relu")){
			gradient = NumJ.copy(d);
			for(int i=0;i<output.length;i++){
				for(int j=0;j<output[0].length;j++){
					if(output[i][j]==0)
						gradient[i][j] = 0;
				}
			}
		}
		else if(function.equals("tanh")){
			try{
				gradient = NumJ.mul(d, NumJ.add(NumJ.neg(NumJ.mul(output, output)), 1.0f));
			}catch(Exception e){
				System.out.println(e);
			}
		}
		else if(function.equals("sigmoid")){
			try{
				gradient = NumJ.mul(d, NumJ.mul(output, NumJ.add(NumJ.neg(output), 1.0f)));
			}catch(Exception e){
				System.out.println(e);
			}
		}
		else if(function.equals("softmax")){
			int batch = d.length;
			if(output.length==d.length && output[0].length==d[0].length)
				gradient = NumJ.copy(d);
			else{
				gradient = new float[output.length][output[0].length];
				for(int i=0;i<d[0].length;i++)
					gradient[i][(int)d[0][i]] = 1;
			}
			try{
				gradient = NumJ.div(NumJ.sub(output, gradient), batch);
			}catch(Exception e){
				System.out.println(e);
			}
		}
		return gradient;
	}

	public static float[][] softmax(float[][] mat) {
		float[][] exp = NumJ.exp(mat);
		float[][] rs = NumJ.sum(exp, 1);
		try{
			exp = NumJ.div(exp, NumJ.transpose(rs));
		} catch(Exception e){
			System.out.println(e);
		}
		return exp;
	}

	public static float[][] sigmoid(float[][] mat) {
		float[][] temp = NumJ.copy(mat);
		try{
			temp = NumJ.reciprocal(NumJ.add(NumJ.exp(NumJ.neg(mat)), 1));	
		} catch(Exception e){
			System.out.println(e);
		}
		return temp;
	}

	public static float[][] tanh(float[][] mat) {
		float[][] temp = NumJ.copy(mat);
		try{
			temp = NumJ.sub(NumJ.mul(sigmoid(NumJ.mul(mat, 2)), 2), 1);
		} catch(Exception e){
			System.out.println(e);
		}
		return temp;
	}

	public static float[][] relu(float[][] mat) {
		float[][] relu = new float[mat.length][mat[0].length];
		for(int i=0;i<mat.length;i++)
			for(int j=0;j<mat[i].length;j++)
				relu[i][j] = mat[i][j]>0 ? mat[i][j] : 0;
		return relu;
	}

	public void updateWeights(float lr, float rs) {
		return;
	}

	public boolean hasWeights() {
		return false;
	}

	public String getName() {
		return name;
	}

	public String getType() {
		return type;
	}
}

class Dense implements Layer {
	private float[][] weights;
	private float[][] bias;
	private float[][] dw;
	private float[][] db;
	private float[][] input;
	private String name;
	private String type;

	public Dense(String name, int input_size, int output_size) {
		this.weights = NumJ.mul(NumJ.rand(input_size, output_size), 0.01f);
		this.bias = NumJ.zeros(output_size);
		this.dw = new float[input_size][output_size];
		this.db = new float[bias.length][bias[0].length];
		this.name = name;
		this.type = "Dense";
	}

	public float[][] forward(float[][] x) {
		input = NumJ.copy(x);
		float[][] temp = NumJ.copy(x);
		try{
			temp = NumJ.add(NumJ.dot(input, weights), bias);
		} catch(Exception e){
			System.out.println(e);
		}
		return temp;
	}

	public float[][] backward(float[][] d) {
		float[][] temp = NumJ.copy(d);
		try{
			dw = NumJ.dot(NumJ.transpose(input), d);
			db = NumJ.sum(d, 0);
			temp = NumJ.dot(d, NumJ.transpose(weights));
		} catch(Exception e){
			System.out.println(e);
		}
		return temp;
	}

	public void updateWeights(float lr, float rs) {
		try{
			dw = NumJ.add(dw, NumJ.mul(weights, rs));
			weights = NumJ.sub(weights, NumJ.mul(dw, lr));
			bias = NumJ.sub(bias, NumJ.mul(db, lr));
		} catch(Exception e) {
			System.out.println(e);
		}
	}

	public boolean hasWeights() {
		return true;
	}

	public String getName() {
		return name;
	}

	public String getType() {
		return type;
	}
}

class NeuralNetwork {
	private float loss;
	private float accuracy;
	private Layer[] layers;
	private float learning_rate;
	private float regularization;

	public float[][] predict(float[][] x) {
		float[][] output = NumJ.copy(x);
		for(Layer l: layers)
			output = l.forward(output);
		return output;
	}

	public float[][] backprop(float[][] t){
		float[][] grad = NumJ.copy(t);
		for(int i=layers.length-1;i>=0;i--)
			grad = layers[i].backward(grad);
		return grad;
	}

	public float meanSquaredError(float[][] label, float[][] pred) {
		try{
			float[][] dif = NumJ.sub(label, pred);
			loss = (0.5f * NumJ.sum(NumJ.mul(dif, dif)));
			return loss;
		}catch(Exception e){
			System.out.println(e);
		}
		return 0;
	}

	public float getAccuracy(float[][] labels, float[][] preds) {
		accuracy = 0;
		int[] pos_preds = NumJ.argmax(preds, 1);
		int[] pos_labels = NumJ.argmax(labels, 1);
		for(int i=0;i<pos_preds.length;i++){
			if(pos_preds[i]==pos_labels[i])
				accuracy += 1.0f;
		}
		accuracy /= pos_labels.length;
		return accuracy;
	}

	public void addLayer(Layer l) {
		if(layers==null){
			layers = new Layer[1];
			layers[0] = l;
			return;
		}
		Layer[] temp = new Layer[layers.length + 1];
		for(int i=0;i<layers.length;i++)
			temp[i] = layers[i];
		temp[temp.length-1] = l;
		layers = temp;
	}

	public void setParameters(float lr, float rs) {
		this.learning_rate = lr;
		this.regularization = rs;
	}

	public void updateWeights() {
		for(Layer l: layers) {
			if(l.hasWeights())
				l.updateWeights(learning_rate, regularization);
		}
	}

	public Layer getLayer(String name) {
		for(Layer l: layers){
			if(name.equals(l.getName()))
				return l;
		}
		return null;
	}
}

public class JDL {
	public static void main(String args[]) {
		NeuralNetwork nn = new NeuralNetwork();
		nn.addLayer(new Dense("Dense1", 3, 4));
		nn.addLayer(new Dense("Dense2", 4, 4));
		nn.addLayer(new Activation("output", "softmax"));
		nn.setParameters(0.01f, 0.001f);
		float[][] x = {{1, 2, 3}};
		float[][] t = {{0, 0, 1, 0}};
		float[][] out;
		for(int i=0;i<10;i++){
			out = nn.predict(x);
			nn.backprop(t);
			System.out.println("\n\nEpoch " + i + "\nLoss: " + nn.meanSquaredError(t, out));
			System.out.println("Accuracy: " + nn.getAccuracy(t, out));
			nn.updateWeights();
		}
	}
}