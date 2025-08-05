public class Edge{
    Node n1; 
    Node n2;
    double weight;

    public Edge(Node n1, Node n2, double weight){
        this.n1 = n1;
        this.n2 = n2;
        this.weight = weight;
    }

    public Node getN1(){
        return n1;
    }
    
    public Node getN2(){
        return n2;
    }

    public double getWeight(){
        return weight;
    }

    public Node getOther(Node n){
        if (n == n1){
            return n2;
        }
        return n1;
    }
    public String toString() {
        return "(" + n1.x + ", " + n1.y + ")    (" + n2.x + ", " + n2.y + ")";
    }
}