import java.util.*;

public class Node{
    int x;
    int y;
    Vector<Edge> adj_vec;
    double source_dist;

    public Node(int x, int y){
        this.x = x;
        this.y = y;
        adj_vec = new Vector<Edge>();
        source_dist = Double.POSITIVE_INFINITY;
    } 
    public int getX(){
        return x;
    }
    
    public int getY(){
        return y;
    }

    
    public Vector getAdjacency(){
        return adj_vec;
    }

    
    public double getDist(){
        return source_dist;
    }

    @Override
     public int hashCode() {
        return x ^ y;
    }

    @Override
    public boolean equals(Object o) {
        Node v = (Node) o;
        return v.x == x && v.y == y;
    }

    public String toString() {
        return "(" + x + ", " + y + ") Source dist:" + source_dist;
    }
}