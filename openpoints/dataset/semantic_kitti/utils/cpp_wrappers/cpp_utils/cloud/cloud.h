//
//
//		0==========================0
//		|    Local feature test    |
//		0==========================0
//
//		version 1.0 : 
//			> 
//
//---------------------------------------------------
//
//		Cloud header
//
//----------------------------------------------------
//
//		Hugues THOMAS - 10/02/2017
//


# pragma once

#include <vector>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <cmath>

#include <time.h>




// Point class
// ***********


class PointXYZ
{
public:

	// Elements
	// ********

	float x, y, z;


	// Methods
	// *******
	
	// Constructor
	PointXYZ() { x = 0; y = 0; z = 0; }
	PointXYZ(float x0, float y0, float z0) { x = x0; y = y0; z = z0; }
	
	// array type accessor
	float operator [] (int i) const
	{
		if (i == 0) return x;
		else if (i == 1) return y;
		else return z;
	}

	// opperations
	float dot(const PointXYZ P) const
	{
		return x * P.x + y * P.y + z * P.z;
	}

	float sq_norm()
	{
		return x*x + y*y + z*z;
	}

	PointXYZ cross(const PointXYZ P) const
	{
		return PointXYZ(y*P.z - z*P.y, z*P.x - x*P.z, x*P.y - y*P.x);
	}	

	PointXYZ& operator+=(const PointXYZ& P)
	{
		x += P.x;
		y += P.y;
		z += P.z;
		return *this;
	}

	PointXYZ& operator-=(const PointXYZ& P)
	{
		x -= P.x;
		y -= P.y;
		z -= P.z;
		return *this;
	}

	PointXYZ& operator*=(const float& a)
	{
		x *= a;
		y *= a;
		z *= a;
		return *this;
	}
};


// Point Opperations
// *****************

inline PointXYZ operator + (const PointXYZ A, const PointXYZ B)
{
	return PointXYZ(A.x + B.x, A.y + B.y, A.z + B.z);
}

inline PointXYZ operator - (const PointXYZ A, const PointXYZ B)
{
	return PointXYZ(A.x - B.x, A.y - B.y, A.z - B.z);
}

inline PointXYZ operator * (const PointXYZ P, const float a)
{
	return PointXYZ(P.x * a, P.y * a, P.z * a);
}

inline PointXYZ operator * (const float a, const PointXYZ P)
{
	return PointXYZ(P.x * a, P.y * a, P.z * a);
}

inline std::ostream& operator << (std::ostream& os, const PointXYZ P)
{
	return os << "[" << P.x << ", " << P.y << ", " << P.z << "]";
}

inline bool operator == (const PointXYZ A, const PointXYZ B)
{
	return A.x == B.x && A.y == B.y && A.z == B.z;
}

inline PointXYZ floor(const PointXYZ P)
{
	return PointXYZ(std::floor(P.x), std::floor(P.y), std::floor(P.z));
}


PointXYZ max_point(std::vector<PointXYZ> points);
PointXYZ min_point(std::vector<PointXYZ> points);











