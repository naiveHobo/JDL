import java.util.concurrent.ThreadLocalRandom;
import NumJ.*;
import Dataset.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JLabel;
import javax.swing.JOptionPane;

import java.awt.*;
import java.awt.event.*;

interface Layer {
	public float[][] forward(float[][] x);
	public float[][] backward(float[][] d);
	public void updateWeights(float lr, float rs);
	public String getName();
	public String getType();
}

// class Dropout implements Layer {
// 	private String name;
// 	private String type;
// 	private float ratio;
// 	private int[][] mask;

// 	public Dropout(String name, float ratio) {
// 		this.name = name;
// 		this.ratio = ratio;
// 		this.type = "Dropout";
// 	}

// 	public float[][] forward(float[][] x) {
// 		float[][] probs
// 	}
// }

class Activation implements Layer {
	private String name;
	private String type;
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

	public String getName() {
		return name;
	}

	public String getType() {
		return type;
	}
}

class NeuralNetwork {
	// private float[][]
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

	public float crossEntropyLoss(float[][] label, float[][] pred) {
		int[] pos = new int[1];
		try{
			pos = NumJ.argmax(label, 1);
		} catch(Exception e){
			System.out.println(e);
		}
		loss = 0;
		for(int i=0;i<pos.length;i++)
			loss += (float) -Math.log(pred[i][pos[i]]);
		loss /= pos.length;
		return loss;
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

	public void updateWeights() {
		for(Layer l: layers)
			l.updateWeights(learning_rate, regularization);
	}

	public void setParameters(float lr, float rs) {
		this.learning_rate = lr;
		this.regularization = rs;
	}

	public void fit(float[][] train_x, float[][] train_y, int epochs, int batch_size) {
		int train_size = train_x.length;
		float[][] batch = new float[batch_size][train_x[0].length];
		float[][] labels = new float[batch_size][train_y[0].length];
		float[][] preds = new float[batch_size][train_y[0].length];
		int[] data_pos = new int[train_size];
		int randomPos;
		float comp = 0;
		int p;

		System.out.println("Commencing training...");

		for(int i=0;i<epochs;i++){
			System.out.println("\n\nEpoch " + (i+1) + ":");
			train_size = train_x.length;
			comp = 0;
			p = 0;
			for(int j=0;j<train_size;j++)
				data_pos[j] = j;
			while(train_size>0){
				batch_size = Math.min(batch_size, train_size);
				if(batch_size==train_size){
					batch = new float[batch_size][train_x[0].length];
					labels = new float[batch_size][train_y[0].length];
				}
				for(int j=0;j<batch_size;j++){
					randomPos = ThreadLocalRandom.current().nextInt(0, train_size);
					batch[j] = train_x[data_pos[randomPos]];
					labels[j] = train_y[data_pos[randomPos]];
					System.arraycopy(data_pos, randomPos+1, data_pos, randomPos, data_pos.length-1-randomPos);
					train_size--;
				}
				preds = predict(batch);
				backprop(labels);
				updateWeights();
				comp += ((batch_size*50.0f)/train_x.length);
				for(;p<comp;p++)
					System.out.print(">");
			}
			System.out.println("\nLoss: " + crossEntropyLoss(labels, preds));
			System.out.println("Accuracy: " + getAccuracy(labels, preds));
		}
	}
}

class Test extends Frame implements WindowListener, ActionListener {
	Button genImg;
	Button genPred;
	JLabel label;
	JLabel text;
	NeuralNetwork nn;
	float[][] img;
	float[][] prediction;
	private String IMG_PATH;

	public Test(String title, NeuralNetwork nn) {
		super(title);
		this.nn = nn;
		img = new float[1][784];
		prediction = new float[1][10];
		setLayout(new FlowLayout());
		addWindowListener(this);
		IMG_PATH = MNIST.randomImage();
		genImg = new Button("Random Image");
		genPred = new Button("Predict");
		add(genImg);
		add(genPred);
		try {
			label = new JLabel(new ImageIcon(ImageIO.read(new File(IMG_PATH))));
			add(label);
		} catch(IOException e){
			System.out.println(e);
		}
		text = new JLabel("\nPrediction: ");
		add(text);

		genImg.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				IMG_PATH = MNIST.randomImage();
				try {
					label.setIcon(new ImageIcon(ImageIO.read(new File(IMG_PATH))));
				} catch(IOException ex){
					System.out.println(ex);
				}
				revalidate();
				repaint();
			}
		});

		genPred.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				try{
					img[0] = MNIST.imageToMatrix(ImageIO.read(new File(IMG_PATH)));
				} catch(IOException ex){
					System.out.println(ex);
				}
				prediction = nn.predict(img);
				text.setText("\nPrediction: " + NumJ.argmax(prediction)[1]);
				revalidate();
				repaint();
			}
		});
	}

	public void windowClosing(WindowEvent e) {
		dispose();
		System.exit(0);
	}

	public void actionPerformed(ActionEvent e) {}
	public void windowOpened(WindowEvent e) {}
	public void windowActivated(WindowEvent e) {}
	public void windowIconified(WindowEvent e) {}
	public void windowDeiconified(WindowEvent e) {}
	public void windowDeactivated(WindowEvent e) {}
	public void windowClosed(WindowEvent e) {}

}

public class JDL {
	public static void main(String args[]) {
		NeuralNetwork nn = new NeuralNetwork();
		nn.addLayer(new Dense("Dense1", 784, 50));
		nn.addLayer(new Dense("Dense2", 50, 10));
		nn.addLayer(new Activation("output", "softmax"));
		nn.setParameters(0.001f, 0.001f);
		float[][] train_x = new float[500][784];
		float[][] train_y = new float[500][10];
		MNIST.loadTrainingData(train_x, train_y, 50);
		System.out.println("Succesfully loaded training data!\n");
		nn.fit(train_x, train_y, 2, 40);
		System.out.println("\nSuccesfully trained the model!\n");
		// float[][] test_x = new float[100][784];
		// float[][] test_y = new float[100][10];
		// MNIST.loadTestData(test_x, test_y, 10);
		// float[][] temp = nn.predict(test_x);
		// System.out.println(nn.getAccuracy(test_y, temp));
		Test myWindow = new Test("Neural Network Tester", nn);
		myWindow.setSize(350,100);
		myWindow.setVisible(true);
	}
}