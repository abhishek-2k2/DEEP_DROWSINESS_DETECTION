#include <iostream>
#include <vector>
#include <queue>
#include <limits.h>
#include <unordered_map>

using namespace std;

// A structure to represent a road (edge) between intersections
struct Road {
    int destination;  // The destination intersection
    int capacity;     // Traffic capacity of the road
    int speed;        // Speed limit on the road
    int time;         // Estimated travel time (calculated later)

    Road(int dest, int cap, int spd) : destination(dest), capacity(cap), speed(spd) {
        time = capacity / speed;  // Basic assumption: time is inversely proportional to speed
    }
};

// A class to represent the city's road network (graph)
class TrafficOptimizationSystem {
public:
    unordered_map<int, vector<Road>> cityGraph;  // Intersection -> Roads (graph structure)
    unordered_map<int, bool> isEmergencyRoute;   // Emergency vehicles' route info

    // Add road between two intersections
    void addRoad(int src, int dest, int capacity, int speed) {
        cityGraph[src].push_back(Road(dest, capacity, speed));
        cityGraph[dest].push_back(Road(src, capacity, speed));  // For undirected roads
    }

    // Dijkstra's algorithm to find the most efficient route (shortest time)
    vector<int> dijkstra(int start, int end) {
        unordered_map<int, int> distances;  // Intersection -> Shortest time to get there
        unordered_map<int, int> previous;   // For path reconstruction
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;  // Min-heap priority queue

        // Initialize all intersections with maximum time
        for (auto& intersection : cityGraph) {
            distances[intersection.first] = INT_MAX;
        }

        // Start at the source intersection
        distances[start] = 0;
        pq.push({0, start});

        while (!pq.empty()) {
            int currTime = pq.top().first;
            int currIntersection = pq.top().second;
            pq.pop();

            if (currIntersection == end) {
                break;  // Reached destination, no need to continue
            }

            // Explore neighbors (connected roads)
            for (const Road& road : cityGraph[currIntersection]) {
                int newTime = currTime + road.time;
                if (newTime < distances[road.destination]) {
                    distances[road.destination] = newTime;
                    previous[road.destination] = currIntersection;
                    pq.push({newTime, road.destination});
                }
            }
        }

        // Reconstruct the path from start to end
        vector<int> path;
        for (int at = end; at != start; at = previous[at]) {
            path.push_back(at);
        }
        path.push_back(start);
        reverse(path.begin(), path.end());  // Reverse the path to get start -> end

        return path;
    }

    // Simulate real-time rerouting based on traffic data (dummy simulation)
    void dynamicReroute(int vehicleCount, int congestionThreshold) {
        // If vehicle count exceeds threshold, reroute traffic
        if (vehicleCount > congestionThreshold) {
            cout << "Traffic congestion detected! Rerouting...\n";
            // Here, you can implement your algorithm to dynamically adjust traffic flow
        }
    }

    // Display the shortest path
    void displayPath(const vector<int>& path) {
        cout << "Shortest Path: ";
        for (int i : path) {
            cout << i << " ";
        }
        cout << endl;
    }
};

// Main function to simulate the traffic flow optimization
int main() {
    TrafficOptimizationSystem system;

    // Add roads (intersections, capacity, speed)
    system.addRoad(1, 2, 100, 60);
    system.addRoad(2, 3, 150, 50);
    system.addRoad(3, 4, 200, 40);
    system.addRoad(4, 5, 120, 70);
    system.addRoad(1, 3, 180, 55);
    system.addRoad(2, 4, 160, 65);

    // Get the shortest path between two intersections (1 to 5)
    vector<int> path = system.dijkstra(1, 5);
    system.displayPath(path);

    // Simulate real-time rerouting based on vehicle count
    system.dynamicReroute(300, 250);  // Simulating 300 vehicles, congestion threshold is 250

    return 0;
}
