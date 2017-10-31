import java.io.*;
import java.util.*;

public class dataload {

	public static void main(String args[])throws IOException {

		String name = args[0];
		File file = new File(name);
		BufferedReader br = new BufferedReader(new FileReader(file));

		String n;
		int k;
		int cnt=0;
		while((n=br.readLine())!=null) {
			cnt++;
			k = Integer.parseInt(n);
			k=k>0?1:0;
			if(cnt!=20)
				System.out.print(k);
			else {
				System.out.println();				
				cnt=0;
			}
		}
	}
}