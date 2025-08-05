import java.lang.reflect.Array;
import java.util.*;
import java.util.concurrent.CyclicBarrier;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Main{
    int n; //Number of nodes in the graph
    int min = 0;
    int max = Integer.MAX_VALUE;
    long sd;
    int degree;
    Random prn; 
    HashMap<Node, Double> graph;
    Vector<Edge> edges;
    Node nodes[];

    // Variables needed for the delta stepping algorithm
    int num_buckets;
    ArrayList<ArrayList<Node>> buckets;    
    double delta;

    // Shared request queue needed to add requests for nodes in another thread's jurisdiction
    List<ArrayList<Request>> shared_q;
    // This arraylist keeps track of the number of nodes in bucket i for all threads when all threads are on bucket i
    ArrayList<Integer> buckets_cap;
    // number of threads (duh)
    int num_threads;
    // Arraylist for keeping track of which node falls in which thread's jurisdiction
    ArrayList<ArrayList<Node>> juris;



    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.println("Enter Number of Nodes for the graph:");
        int n = sc.nextInt();
        // int num_threads = sc.nextInt();
        Main ds = new Main(n, 1);
        Main djk = new Main(n, 1);
        long djk_avg = 0;
        long pds_avg = 0;
        long ds_avg = 0;

        long[] pds_avgs = new long[4];
        
        System.out.println("Started Delta Stepping");
        
        for (int i = 0; i < 3; i++){
            long ds_start = System.currentTimeMillis();

            ds.sequential_delta_stepping();

            long ds_end = System.currentTimeMillis();
            ds_avg += ds_end - ds_start;
        }
        ds_avg = ds_avg/3;        
        
        System.out.println("Started Djikstra's");
        for (int i = 0; i < 3; i++){
            long djk_start = System.currentTimeMillis();

            djk.update_paths();
            
            long djk_end = System.currentTimeMillis();
            djk_avg += djk_end - djk_start;
        }
        djk_avg = djk_avg/3;
        
        
        // Boolean err = false;
            int j = 0;
            for (int tc = 2; tc <= 16; tc = tc*2){
                System.err.println("Started Parallel Delta Stepping for thread count " + tc);
                for (int i = 0; i < 3; i++){
                    
                    Main pds = new Main(n, tc);
                    long pds_start = System.currentTimeMillis();

                    pds.parallel_delta_stepping();
                    
                    long pds_end = System.currentTimeMillis();
                    pds_avg += pds_end - pds_start;
                }
                pds_avgs[j] = pds_avg/3;
                j++;
            }
            
        System.out.println("Djikstra's 10 run average time: " + djk_avg + " millis");
        System.out.println("Sequential Delta Stepping 10 run average time: " + ds_avg + " millis");
        for (int i = 0; i < pds_avgs.length; i++){
            System.out.println("Parallel Delta Stepping 10 run average for thread count " + Math.pow(2, i+1) + ": " + pds_avgs[i] + "millis");
        }
    }

    Boolean empty; 
    Boolean done;
    Boolean shared_req_empty;
    CyclicBarrier barrier1;
    CyclicBarrier barrier2;
    CyclicBarrier barrier3;
    CyclicBarrier barrier4;

    public void parallel_delta_stepping(){
        done = true;
        empty = true;
        int range = n/num_threads;
        
        shared_q = Collections.synchronizedList(new ArrayList<>(num_threads));

        buckets_cap = new ArrayList<Integer>(this.num_threads);

        juris = new ArrayList<ArrayList<Node>>(num_threads);

        for (int i = 0; i < num_threads; i++){
            shared_q.add(new ArrayList<Request>());
            juris.add(new ArrayList<Node>());
            buckets_cap.add(0);
        }

        for (int i = 0; i < num_threads; i++){
            // If the number of nodes is not evenly divisible by the number of threads then the last thread handles the left over nodes
            int num_extras = (i == num_threads - 1)? n % num_threads : 0;
            // Adding nodes to the juris arraylist to define jurisdiction for each thread
            for (int j = i*range; j < i*range + range + num_extras; j++){
                juris.get(i).add(nodes[j]);
            }
        }

        // for (int i = 0; i < juris.size(); i++){
        //     System.out.println(juris.get(i).size());
        // }
        barrier1 = new CyclicBarrier(num_threads, new Runnable(){
                                                        public void run(){
                                                            for (int i=0; i<buckets_cap.size(); i++){
                                                                if (buckets_cap.get(i) > 0) done = false;
                                                            }
                                                        }
        });

        barrier2 = new CyclicBarrier(num_threads, new Runnable(){
                                                            public void run(){
                                                                for (int i=0; i<shared_q.size(); i++){
                                                                    if (shared_q.get(i).size() > 0) {
                                                                        shared_req_empty = false;
                                                                        return;}
                                                                }
                                                            }
        });

        // barrier2 = new CyclicBarrier(num_threads);

        barrier3 = new CyclicBarrier(num_threads, new Runnable(){
                                                            public void run(){
                                                                for(int i = 0; i < buckets_cap.size(); i++){
                                                                    if(buckets_cap.get(i) > 0) empty = false;
                                                                }
                                                            }
        });

        barrier4 = new CyclicBarrier(num_threads);

        Thread threads[] = new Thread[num_threads];
        for (int i = 0; i < num_threads; i++){
            int num_extras = (i == num_threads - 1)? n % num_threads : 0;
            threads[i] = new Thread(new parallel_ds_thread(i));
            threads[i].start(); 
        }
        try {
            for(Thread t : threads){
                t.join();
            }
        } catch (Exception e) { e.printStackTrace(); }
    }

    public class parallel_ds_thread implements Runnable{
        // Current thread's index
        int t_index;
        // Current thread's individual bucket
        ArrayList<ArrayList<Node>> t_bucket;
        // Start and end indices for the range of nodes to be handled by the given thread
        int n_start_ind;
        int n_end_ind;

        public parallel_ds_thread(int t_index){
            // this.n_start_ind = n_start_ind;
            // this.n_end_ind = n_end_ind;
            this.t_index = t_index;
            t_bucket = new ArrayList<ArrayList<Node>>(num_buckets);
            for (int i=0; i<num_buckets; i++){
                t_bucket.add(new ArrayList<Node>());
            }
        }

        public int find_juris_tind(Node n){
            for (int i = 0; i < num_threads; i++){
                if (juris.get(i).contains(n)){
                    return i;
                }
            }
            // This should never run (because all nodes are in the jurisdiction of some thread)
            // But the compiler doesn't get it so have to add this redundant return
            return -1;
        }

        @Override
        public void run(){
            t_bucket = relax(nodes[0], 0.0, t_bucket);
            
            while(true){
                int j;
                for (j = 0; j < num_buckets; j++){
                    // for a given buckets all threads will update their indiv bucket size into the bucket_cap list
                    // if for bucket number j any thread has more nodes, we will break out and all threads will work on that bucket
                    done = true;
                    try{
                        buckets_cap.set(t_index, t_bucket.get(j).size());
                        barrier1.await();
                    }catch(Exception ex){ex.printStackTrace();}
                    if (!done){
                        break;
                    }
                }
                // Did we break out of for loop prematurely or because we went over all buckets for all threads and found nothing to work on
                // If latter then we are done and we break out of the loop
                if (j >= num_buckets){
                    break;
                }

                ArrayList<Node> r = new ArrayList<Node>(t_bucket.get(j).size());
                ArrayList<Request> requests;

                while(true){
                    requests = find_requests(t_bucket.get(j), "light");
                    r.addAll(t_bucket.get(j));
                    t_bucket.get(j).clear();
                    for (Request req : requests){
                        int t_ind = find_juris_tind(req.n);
                        if (t_ind == this.t_index){
                            t_bucket = relax(req.n, req.dist, t_bucket);
                        }else{
                            shared_q.get(t_ind).add(req);
                        }
                    }
                    shared_req_empty = true;
                    try {
                        barrier2.await();
                    } catch (Exception e) { e.printStackTrace(); }
                    // if (shared_req_empty) break;
                    if (!shared_q.get(t_index).isEmpty()){
                        // System.out.println("Before relax: " + shared_q.get(t_index).size());
                        for (Request s_req : shared_q.get(t_index)){
                            t_bucket = relax(s_req.n, s_req.dist, t_bucket);
                        }
                        // System.out.println("After relax: " + shared_q.get(t_index).size());
                        shared_q.get(t_index).clear();
                    }
                    empty = true;
                    try {
                        buckets_cap.set(t_index, t_bucket.get(j).size());
                        barrier3.await();
                    } catch (Exception e) { e.printStackTrace(); }

                    if(empty) {
                        // System.out.println("Can break out of inner loop");  
                        break;  
                    }
                }
                requests = find_requests(r, "heavy");
                for (Request req : requests){
                    int t_ind = find_juris_tind(req.n);
                    if (t_ind == this.t_index){
                        t_bucket = relax(req.n, req.dist, t_bucket);
                    }else{
                        shared_q.get(t_ind).add(req);
                    }
                }
                try {
                    barrier4.await();
                } catch (Exception e) { e.printStackTrace(); }
                if (!shared_q.get(t_index).isEmpty()){
                    for(Request s_req : shared_q.get(t_index)){
                        t_bucket = relax(s_req.n, s_req.dist, t_bucket);
                    }
                    shared_q.get(t_index).clear();
                }
            }
        }
    }

    // Adapted from the pseudo code delta stepping algorithm available on the wikipedia page: https://en.wikipedia.org/wiki/Parallel_single-source_shortest_path_algorithm 
    public void sequential_delta_stepping(){
        for (int i = 0; i < num_buckets; i++){
            buckets.add(new ArrayList<>());
        }
        // Since the constructor for all nodes sets their tentative distances to inf anyways I'll skip that
        buckets = relax(nodes[0], 0.0, buckets);

        while (true){
            int j;
            for (j = 0; j < num_buckets; j++){
                if (!buckets.get(j).isEmpty()){
                    break;
                }
            }
            // If j >= num_buckets then the loop traversed over all buckets and found them all empty
            if (j >= num_buckets){
                // System.out.println(" j " + j);
                // If there is no more buckets with nodes in them then we are done
                break; // Breaks out of the while(true) loop
            }
            // Arraylist for keeping track of deleted nodes
            ArrayList<Node> r = new ArrayList<>(buckets.get(j).size());
            ArrayList<Request> requests;
            while (!buckets.get(j).isEmpty()){
                // Find the light requests
                requests = find_requests(buckets.get(j), "light");
                // Remember deleted nodes
                r.addAll(buckets.get(j));
                // Clear the bucket of all nodes
                buckets.get(j).clear();

                // Relax the light request queue
                buckets = relax_requests(requests, buckets);
            }
            // Create requests for heavy edges
            requests = find_requests(r, "heavy");

            // Finally relax the heavy requests
            buckets = relax_requests(requests, buckets);
        }
    }

    public class Request{
        Node n;
        Double dist;
        public Request(Node n, Double dist){
            this.n = n;
            this.dist = dist;
        }
    }

    public ArrayList<Request> find_requests(ArrayList<Node> node_set, String kind){
        ArrayList<Request> requests = new ArrayList<Request>();
        if (kind.equals("light")){
            // Go over each node in the given node set
            for (Node n : node_set){
                // Go over all out going edges of each node and find the light edges
                for (Edge e : n.adj_vec){
                    // Light edges are defined as edges with weight at most delta
                    if (e.weight <= delta){
                        requests.add(new Request(e.getOther(n), n.source_dist + e.getWeight()));
                    }
                }
            }
        }

         else{
            // Go over each node in the given node set
            for (Node n : node_set){
                // Go over all out going edges of each node and find the heavy edges
                for (Edge e : n.adj_vec){
                    // Heavy edges are defined as edges with weight more than delta
                    if (e.weight > delta){
                        requests.add(new Request(e.getOther(n), n.source_dist + e.getWeight()));
                    }
                }
            }
        }
        return requests;
    }

    public ArrayList<ArrayList<Node>> relax_requests(ArrayList<Request> requests, ArrayList<ArrayList<Node>> buckets){   
        for (Request req : requests){
            buckets = relax(req.n, req.dist, buckets);
        }
        return buckets;
    }

    public ArrayList<ArrayList<Node>> relax(Node n, Double dist, ArrayList<ArrayList<Node>> buckets){
        // If the new found distance from source is less than the old dist then we have a new bucket for the node
        if (dist < n.source_dist){
            // Remove node from bucket's source_dist/delta bucket
            int j = (int) (n.source_dist/delta % num_buckets);
            buckets.get(j).remove(n);
            
            // Insert node into bucket's dist/delta bucket instead
            j = (int)(dist/delta % num_buckets);
            buckets.get(j).add(n);

            // Change the node's source dist to the new found dist now
            n.source_dist = dist;
            for (Node nod : nodes){
                if (nod.equals(n)){
                    nod.source_dist = dist;
                }
            }
        }
        return buckets;
    }

    public void update_paths() {
    	//runs djikstras on all nodes in graph
    	//vars: graph, edges, nodes
		dijkstra(nodes[0],nodes, edges);
		// for(int i = 0; i < nodes.length;i++) {
		// 	System.out.println(nodes[i].toString());
		// }
		// for(int i = 0; i < edges.size();i++) {
		// 	System.out.println(edges.get(i).toString());
		// }
    }
    //pseudo code from wikipedia.com
//  1   function Dijkstra(Graph, source):
//  	2       create vertex priority queue Q
//  	3
//  	4       dist[source] = 0                          // Initialization
//  	5       Q.add_with_priority(source, 0)            // associated priority equals dist[·]
//  	6
//  	7       for each vertex v in Graph.Vertices:
//  	8           if v != source
//  	9               prev[v] = UNDEFINED               // Predecessor of v
//  	10              dist[v] = INFINITY                // Unknown distance from source to v
//  	11              Q.add_with_priority(v, INFINITY)
//  	12
//  	13
//  	14      while Q is not empty:                     // The main loop
//  	15          u = Q.extract_min()                   // Remove and return best vertex
//  	16          for each neighbor v of u:             // Go through all v neighbors of u
//  	17              alt = dist[u] + Graph.Edges(u, v)
//  	18              if alt < dist[v]:
//  	19                  prev[v] = u
//  	20                  dist[v] = alt
//  	21                  Q.decrease_priority(v, alt)
//  	22
//  	23      return dist, prev
    private boolean dijkstra(Node src, Node[] nodes,Vector<Edge> edges) {
    	PriorityQueue<Node> pq = new PriorityQueue<Node>(nodes.length, new NodeComparator());
    	src.source_dist = 0.0;
    	pq.add(src);
    	Node u;
    	Node cur;
    	Double alt;
    	while(!pq.isEmpty()) {
    		u = pq.poll();

//    		System.out.println("u: " + u.toString());
    		for(Edge e : u.adj_vec) {
//    			System.out.println("e: " + e.toString());
				cur = e.getOther(u);
				alt = u.getDist() + e.getWeight();
				if(alt < cur.source_dist) {
					cur.pred = e;
					cur.source_dist = alt;
					pq.remove(cur);
					pq.add(cur);
				}
			}
    	}
    	return true;
    }
    class NodeComparator implements Comparator<Node>{
         public int compare(Node n1, Node n2) {
        	 // compares node source distances
        	 if (n1.source_dist == n2.source_dist)
        		 return 0;
        	 else if(n1.source_dist < n2.source_dist)
        		 return -1;
        	 else
        		 return 1;
         }
   }


    public Main(int n, int num_threads){
        this.n = n;
        graph = new HashMap<Node, Double>(n);
        degree = 5;
        prn = new Random();
        nodes = new Node[n];
        edges = new Vector<Edge>();
        prn.setSeed(1); // idk what the seed should be for this
        num_buckets = 10;
        buckets = new ArrayList<ArrayList<Node>>(num_buckets);
        delta = max / degree;

        // Initializing stuff for the parallel ds version
        this.num_threads = num_threads;
        graph_maker(n);
    }

    // Generating the graph manually; needed to generate the graph manually because the ready made graphs on the SNAP website were too small
    // Adapted the code from professor Michael Scott's implementation of graph generation (tried doing it myself but it was too complicated to make a weighted graph)

    class CheckerBoard {
        private Object[][] cb;
        @SuppressWarnings("unchecked")
        public Vector<Node> get(int i, int j) {
            return (Vector<Node>)(cb[i][j]);
        }
        public CheckerBoard (int k) {
            cb = new Object[k][k];
            // Really Vector<Vertex>, but Java erasure makes that illegal.
            for (int i = 0; i < k; ++i) {
                for (int j = 0; j < k; ++j) {
                    cb[i][j] = new Vector<Node>();
                }
            }
        }
    }

    private int euclideanDistance(Node v1, Node v2) {
        double xDiff = v1.x - v2.x;
        double yDiff = v1.y - v2.y;
        return (int) Math.sqrt(xDiff * xDiff + yDiff * yDiff);
    }

    // Function to create a graph given number of nodes
    public void graph_maker(int n){

        final int k = (int) (Math.sqrt((double)n/(double)degree) * 3 / 2);
        final int sw = (int) Math.ceil((double)max/(double)k);     // square width;
        CheckerBoard cb = new CheckerBoard(k);

        for (int i = 0; i < n; i++){
            Node node;
            int x;
            int y;
            while (true) { 
                x = Math.abs(prn.nextInt()) % max;
                y = Math.abs(prn.nextInt()) % max;
                node = new Node(x, y);
                if (!graph.containsKey(node)){
                    break;
                }        
            }
            graph.put(node, node.source_dist);
            nodes[i] = node;
            cb.get(x/sw, y/sw).add(node);
        }
        // nodes[0].source_dist = 0;

        for (Node node : nodes) {
            int xb = node.x / sw;
            int yb = node.y / sw;
            // Find 3x3 area from which to draw neighbors.
            int xl;  int xh;
            int yl;  int yh;
            if (k < 3) {
                xl = yl = 0;
                xh = yh = k-1;
            } else {
                xl = (xb == 0) ? 0 : ((xb == k-1) ? k-3 : (xb-1));
                xh = (xb == 0) ? 2 : ((xb == k-1) ? k-1 : (xb+1));
                yl = (yb == 0) ? 0 : ((yb == k-1) ? k-3 : (yb-1));
                yh = (yb == 0) ? 2 : ((yb == k-1) ? k-1 : (yb+1));
            }
            for (int i = xl; i <= xh; ++i) {
                for (int j = yl; j <= yh; ++j) {
                    for (Node u : cb.get(i, j)) {
                        if (node.hashCode() < u.hashCode()
                                // Only choose edge from one end --
                                // avoid self-loops and doubled edges.
                                && prn.nextInt() % 4 == 0) {
                            // Invent a weight.
                            int dist = euclideanDistance(u, node);
                            int weight = (int) (dist);
                            // Pick u as neighbor.
                            Edge e = new Edge(u, node, weight);
                            u.adj_vec.add(e);
                            node.adj_vec.add(e);
                            edges.add(e);
                        }
                    }
                }
            }
        }

    }
}