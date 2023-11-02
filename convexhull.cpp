#include <stdio.h>
#include <algorithm>

struct Vec2d {
	int x, y;
};

inline bool operator < (const Vec2d &a, const Vec2d &b) {
	return a.y < b.y || (a.y == b.y && a.x < b.x);
}

inline Vec2d operator - (const Vec2d &a, const Vec2d &b) {
	return (Vec2d) {a.x - b.x, a.y - b.y};
}

inline long long cross(const Vec2d &a, const Vec2d &b) {
	return 1LL * a.x * b.y - 1LL * a.y * b.x;
}

inline long long cross(const Vec2d &a, const Vec2d &b, const Vec2d &c) {
	return cross(b - a, c - a);
}

inline long long dist2(const Vec2d &a) {
	return 1LL * a.x * a.x + 1LL * a.y * a.y;
}

// ==============================

// Graham Scan Algorithm

void graham(int n, const Vec2d *points, int &m, int *extreme_points) {
	// Find the lowest-then-leftmost (LTL) point
	int LTL = 0;
	for (int i = 1; i < n; i++) {
		if (points[i] < points[LTL]) {
			LTL = i;
		}
	}
	
	// Sort all points by polar angles
	int *ids = new int[n];
	for (int i = 0; i < n; i++) ids[i] = i;
	std::swap(ids[0], ids[LTL]);
	std::sort(ids + 1, ids + n, [LTL, points](int i, int j) {
		long long c = cross(points[LTL], points[i], points[j]);
		return c > 0 || (c == 0 &&
			dist2(points[i] - points[LTL]) < dist2(points[j] - points[LTL]));
		// To assure the second extreme point is correct
	});
	
	// Find all extreme points using monotonic stack
	int *stack = new int[n];
	int top = 1;
	stack[0] = LTL;
	for (int i = 1; i < n; i++) {
		int id = ids[i];
		while (top >= 2 && cross(points[stack[top - 2]],
			points[stack[top - 1]], points[id]) <= 0) {
			// NO three points on a same line
			top--;
		}
		stack[top++] = id;
	}
	
	// Find all extreme points on the edge of the convex hull
	m = 1;
	extreme_points[0] = LTL;
	int p = 1;
	for (int i = 1; i < n; i++) {
		int id = ids[i];
		while (p + 1 < top - 1 &&
			cross(points[LTL], points[stack[p + 1]], points[id]) > 0) {
			++p;
		}
		if (cross(points[stack[p]], points[stack[p + 1]], points[id]) == 0) {
			extreme_points[m++] = id;
			continue;
		}
		if (p == 1 && cross(points[LTL], points[stack[1]], points[id]) == 0) {
			extreme_points[m++] = id;
			continue;
		}
		if (p + 1 == top - 1 &&
			cross(points[LTL], points[stack[top - 1]], points[id]) == 0) {
			extreme_points[m++] = id;
		}
	}
	
	delete[] stack;
}

// =====================

// IO

namespace IO {
	const int BUFSIZE = 1 << 15;
	char *buf = new char[BUFSIZE + 1];
	const char *buf_end = buf + BUFSIZE;
	const char *cur = buf_end;
	
	inline char get_char() {
		if (cur == buf_end) {
			cur = buf;
			unsigned n = fread(buf, 1, BUFSIZE, stdin);
			buf[n] = 0;
		}
		return *(cur++);
	}
	
	inline int get_int() {
		char c = get_char();
		while (c <= 32) c = get_char();
		int f = 1;
		if (c == '-') f = -1, c = get_char();
		int x = 0;
		while (c > 32) x = x * 10 + c - 48, c = get_char();
		return x * f;
	}
}

// =====================

void read_data(int n, Vec2d *points) {
	for (int i = 0; i < n; i++) {
		points[i].x = IO::get_int();
		points[i].y = IO::get_int();
	}
}

int calc_ans(int m, const int *extreme_points) {
	int ans = m;
	const int MOD = 1000000007;
	for (int i = 0; i < m; i++) {
		ans = 1LL * ans * (extreme_points[i] + 1) % MOD;
	}
	return ans;
}

int main() {
	int n;
	n = IO::get_int();
	
	Vec2d *points = new Vec2d[n];
	read_data(n, points);
	
	int *extreme_points = new int[n];
	int m;
	graham(n, points, m, extreme_points);
	
	int ans = calc_ans(m, extreme_points);
	printf("%d\n", ans);
	
	delete[] extreme_points;
	delete[] points;
}
