#pragma once

// #define POSTPROCSHADERS

#define MAXJOBTHREADS	16
#define MAXJOBS			512

#include "math.h"
#include "stdlib.h"
#include "emmintrin.h"
#include "stdio.h"
#include "windows.h"
#include "float.h"


#ifndef REALINLINE
	#ifdef _MSC_VER
		#define REALINLINE __forceinline
	#else
		#define REALINLINE inline
	#endif
#endif


inline float Rand( float a_Range ) { return ((float)rand() / RAND_MAX) * a_Range; }
int filesize( FILE* f );
#define MALLOC64(x) _aligned_malloc(x,64)
#define FREE64(x) _aligned_free(x)
#define MALLOC16(x) _aligned_malloc(x,16)
#define FREE16(x) _aligned_free(x)



#ifndef __INTEL_COMPILER
#define restrict
#endif

#define MIN(a,b) (((a)>(b))?(b):(a))
#define MAX(a,b) (((a)>(b))?(a):(b))

#define _fabs	fabsf
#define _cos	cosf
#define _sin	sinf
#define _acos	acosf
#define _floor	floorf
#define _ceil	ceilf
#define _sqrt	sqrtf
#define _pow	powf
#define _exp	expf

#define CROSS(A,B)		vector3(A.y*B.z-A.z*B.y,A.z*B.x-A.x*B.z,A.x*B.y-A.y*B.x)
#define DOT(A,B)		(A.x*B.x+A.y*B.y+A.z*B.z)
#define ABSDOT(A,B)		(fabs((A.x*B.x+A.y*B.y+A.z*B.z)))
#define NORMALIZE(A)	{float l=1/_sqrt(A.x*A.x+A.y*A.y+A.z*A.z);A.x*=l;A.y*=l;A.z*=l;}
#define CNORMALIZE(A)	{float l=1/_sqrt(A.r*A.r+A.g*A.g+A.b*A.b);A.r*=l;A.g*=l;A.b*=l;}
#define LENGTH(A)		(_sqrt(A.x*A.x+A.y*A.y+A.z*A.z))
#define SQRLENGTH(A)	(A.x*A.x+A.y*A.y+A.z*A.z)
#define SQRDISTANCE(A,B) ((A.x-B.x)*(A.x-B.x)+(A.y-B.y)*(A.y-B.y)+(A.z-B.z)*(A.z-B.z))

#define PI				3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798f

#define PREFETCH(x) _mm_prefetch((const char*)(x),_MM_HINT_T0)
#define PREFETCH_ONCE(x) _mm_prefetch((const char*)(x),_MM_HINT_NTA)
#define PREFETCH_WRITE(x) _m_prefetchw((void*)(x))

#define loadss(mem)			_mm_load_ss((const float*const)(mem))
#define broadcastps(ps)		_mm_shuffle_ps((ps),(ps), 0)
#define broadcastss(ss)		broadcastps(loadss((ss)))

const float ROUNDING_ERROR_f32 = 0.000001f;

//! returns if a equals zero, taking rounding errors into account
inline bool iszero(const float a, const float tolerance = ROUNDING_ERROR_f32)
{
	return fabsf(a) <= tolerance;
}

//! returns if a equals b, taking possible rounding errors into account
inline bool fequals(const float a, const float b, const float tolerance = ROUNDING_ERROR_f32)
{
	return (a + tolerance >= b) && (a - tolerance <= b);
}

//! clamps a value between low and high
template <class T>
inline const T clamp(const T& value, const T& low, const T& high)
{
	return min(max(value, low), high);
}

// calculate: 1 / sqrt ( x )
REALINLINE float reciprocal_squareroot(const float x)
{
	return 1.0f / sqrt(x);
}

// calculate: 1 / x
REALINLINE float reciprocal(const float f)
{
	return 1.f / f;
}



class vector3p
{
public:
	vector3p() {};
	vector3p( float a_X, float a_Y, float a_Z ) : x( a_X ), y( a_Y ), z( a_Z ) {}
	void Set( float a_X, float a_Y, float a_Z ) { x = a_X; y = a_Y; z = a_Z; }
	void Normalize() { float l = 1.0f / Length(); x *= l; y *= l; z *= l; }
	float Length() const { return (float)sqrt( x * x + y * y + z * z ); }
	float SqrLength() const { return x * x + y * y + z * z; }
	float Dot( vector3p a_V ) const { return x * a_V.x + y * a_V.y + z * a_V.z; }
	void operator += ( const vector3p& a_V ) { x += a_V.x; y += a_V.y; z += a_V.z; }
	void operator += ( vector3p* a_V ) { x += a_V->x; y += a_V->y; z += a_V->z; }
	void operator -= ( const vector3p& a_V ) { x -= a_V.x; y -= a_V.y; z -= a_V.z; }
	void operator -= ( vector3p* a_V ) { x -= a_V->x; y -= a_V->y; z -= a_V->z; }
	void operator *= ( const float f ) { x *= f; y *= f; z *= f; }
	void operator *= ( const vector3p& a_V ) { x *= a_V.x; y *= a_V.y; z *= a_V.z; }
	void operator *= ( vector3p* a_V ) { x *= a_V->x; y *= a_V->y; z *= a_V->z; }
	float& operator [] ( int a_N ) { return cell[a_N]; }
	vector3p operator- () const { return vector3p( -x, -y, -z ); }
	friend vector3p operator + ( const vector3p& v1, const vector3p& v2 ) { return vector3p( v1.x + v2.x, v1.y + v2.y, v1.z + v2.z ); }
	friend vector3p operator + ( const vector3p& v1, vector3p* v2 ) { return vector3p( v1.x + v2->x, v1.y + v2->y, v1.z + v2->z ); }
	friend vector3p operator - ( const vector3p& v1, const vector3p& v2 ) { return vector3p( v1.x - v2.x, v1.y - v2.y, v1.z - v2.z ); }
	friend vector3p operator - ( const vector3p& v1, vector3p* v2 ) { return vector3p( v1.x - v2->x, v1.y - v2->y, v1.z - v2->z ); }
	friend vector3p operator - ( const vector3p* v1, vector3p& v2 ) { return vector3p( v1->x - v2.x, v1->y - v2.y, v1->z - v2.z ); }
	friend vector3p operator * ( const vector3p& v, const float f ) { return vector3p( v.x * f, v.y * f, v.z * f ); }
	friend vector3p operator * ( const vector3p& v1, const vector3p& v2 ) { return vector3p( v1.x * v2.x, v1.y * v2.y, v1.z * v2.z ); }
	friend vector3p operator * ( const float f, const vector3p& v ) { return vector3p( v.x * f, v.y * f, v.z * f ); }
	friend vector3p operator / ( const vector3p& v, const float f ) { return vector3p( v.x / f, v.y / f, v.z / f ); }
	friend vector3p operator / ( const vector3p& v1, const vector3p& v2 ) { return vector3p( v1.x / v2.x, v1.y / v2.y, v1.z / v2.z ); }
	friend vector3p operator / ( const float f, const vector3p& v ) { return vector3p( v.x / f, v.y / f, v.z / f ); }
	union
	{
		struct { float x, y, z; };
		struct { float r, g, b; };
		struct { float cell[3]; };
		struct { int ix, iy, iz; };
	};
};

class vector3
{
public:
	vector3() : w( 1.0f ) {};
	vector3( float a_X, float a_Y, float a_Z ) : x( a_X ), y( a_Y ), z( a_Z ), w( 1.0f ) {};
	vector3(const float xyz[3]) : x(xyz[0]), y(xyz[1]), z(xyz[2]), w(1.0f) {};
	vector3( const vector3p& a_V ) : x( a_V.x ), y( a_V.y ), z( a_V.z ) {};
	void Set( float a_X, float a_Y, float a_Z ) { x = a_X; y = a_Y; z = a_Z; }
	void Normalize() { float l = 1.0f / Length(); x *= l; y *= l; z *= l; }
	float Length() const { return (float)sqrt( x * x + y * y + z * z ); }
	const int MinAxis() const { return (fabs(x)<=fabs(y)?((fabs(x)<=fabs(z))?0:2):((fabs(y)<=fabs(z))?1:2)); }
	const vector3 Perpendicular() const
	{
		int ma = MinAxis(); 
		if (ma == 0) return vector3( 0, z, -y ); else if (ma == 1) return vector3( z, 0, -x ); else return vector3( y, -x, 0 );
	}
	float SqrLength() const { return x * x + y * y + z * z; }
	float Dot( const vector3& a_V ) const { return x * a_V.x + y * a_V.y + z * a_V.z; }
	float Dot( const vector3p& a_V ) const { return x * a_V.x + y * a_V.y + z * a_V.z; }
#if 1
	vector3 Cross( const vector3& v ) const { return vector3( y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x ); }
#else
	vector3 Cross( vector3 v ) const 
	{ 
		union { __m128 xmm0; float xmm[4]; };
		xmm0 = xyzw;
		__m128 xmm1 = v.xyzw;
		__m128 xmm2 = xmm0;
		__m128 xmm3 = xmm1;
		xmm0 = _mm_shuffle_ps(xmm0, xmm0, 0xD8);
		xmm1 = _mm_shuffle_ps(xmm1, xmm1, 0xE1);
		xmm0 = _mm_mul_ps(xmm0, xmm1);
		xmm2 = _mm_shuffle_ps(xmm2, xmm2, 0xE1);
		xmm3 = _mm_shuffle_ps(xmm3, xmm3, 0xD8);
		xmm2 = _mm_mul_ps(xmm2, xmm3);
		xmm0 = _mm_sub_ps(xmm0, xmm2);
		return vector3( xmm[2], xmm[1], xmm[0] );
	}
#endif
	float Min() const { return MIN( MIN( x, y ), z ); }
	float Max() const { return MAX( MAX( x, y ), z ); }
	vector3 Min( const vector3& a_V ) const { return vector3( MIN( a_V.x, x ) , MIN( a_V.y, y ) , MIN( a_V.z, z ) ); }
	vector3 Max( const vector3& a_V ) const { return vector3( MAX( a_V.x, x ) , MAX( a_V.y, y ) , MAX( a_V.z, z ) ); }
	void operator += ( const vector3& a_V ) { x += a_V.x; y += a_V.y; z += a_V.z; }
	void operator += ( vector3* a_V ) { x += a_V->x; y += a_V->y; z += a_V->z; }
	void operator -= ( const vector3& a_V ) { x -= a_V.x; y -= a_V.y; z -= a_V.z; }
	void operator -= ( vector3* a_V ) { x -= a_V->x; y -= a_V->y; z -= a_V->z; }
	void operator *= ( const float f ) { x *= f; y *= f; z *= f; }
	void operator *= ( const vector3& a_V ) { x *= a_V.x; y *= a_V.y; z *= a_V.z; }
	void operator *= ( const vector3p& a_V ) { x *= a_V.x; y *= a_V.y; z *= a_V.z; }
	void operator *= ( vector3* a_V ) { x *= a_V->x; y *= a_V->y; z *= a_V->z; }
	operator float* () { return &x; }
	operator const float* () const { return &x; }
	float& operator [] ( int a_N ) { return cell[a_N]; }
	vector3 operator- () const { return vector3( -x, -y, -z ); }
	friend vector3 operator + ( const vector3& v1, const vector3& v2 ) { return vector3( v1.x + v2.x, v1.y + v2.y, v1.z + v2.z ); }
	friend vector3 operator + ( const vector3& v1, vector3* v2 ) { return vector3( v1.x + v2->x, v1.y + v2->y, v1.z + v2->z ); }
	friend vector3 operator - ( const vector3& v1, const vector3& v2 ) { return vector3( v1.x - v2.x, v1.y - v2.y, v1.z - v2.z ); }
	friend vector3 operator - ( const vector3& v1, vector3* v2 ) { return vector3( v1.x - v2->x, v1.y - v2->y, v1.z - v2->z ); }
	friend vector3 operator - ( const vector3* v1, vector3& v2 ) { return vector3( v1->x - v2.x, v1->y - v2.y, v1->z - v2.z ); }
	// friend vector3 operator - ( const vector3* v1, vector3* v2 ) { return vector3( v1->x - v2->x, v1->y - v2->y, v1->z - v2->z ); }
	friend vector3 operator ^ ( const vector3& A, const vector3& B ) { return vector3(A.y*B.z-A.z*B.y,A.z*B.x-A.x*B.z,A.x*B.y-A.y*B.x); }
	friend vector3 operator ^ ( const vector3& A, vector3* B ) { return vector3(A.y*B->z-A.z*B->y,A.z*B->x-A.x*B->z,A.x*B->y-A.y*B->x); }
	friend vector3 operator * ( const vector3& v, const float f ) { return vector3( v.x * f, v.y * f, v.z * f ); }
	friend vector3 operator * ( const vector3& v1, const vector3& v2 ) { return vector3( v1.x * v2.x, v1.y * v2.y, v1.z * v2.z ); }
	friend vector3 operator * ( const float f, const vector3& v ) { return vector3( v.x * f, v.y * f, v.z * f ); }
	friend vector3 operator / ( const vector3& v, const float f ) { return vector3( v.x / f, v.y / f, v.z / f ); }
	friend vector3 operator / ( const vector3& v1, const vector3& v2 ) { return vector3( v1.x / v2.x, v1.y / v2.y, v1.z / v2.z ); }
	friend vector3 operator / ( const float f, const vector3& v ) { return vector3( v.x / f, v.y / f, v.z / f ); }
	union
	{
		struct { float x, y, z, w; };
		struct { float r, g, b, a; };
		struct { float cell[4]; };
		struct { int ix, iy, iz, iw; };
		struct { __m128 xyzw; };
	};
};

class aabb
{
public:
	aabb()
	{
		min = vector3( FLT_MAX, FLT_MAX, FLT_MAX );
		max = vector3(-FLT_MAX,-FLT_MAX,-FLT_MAX );
	}
	aabb( const vector3& bmin, const vector3& bmax )
	{
		min = bmin;
		max = bmax;
	}
	aabb( const vector3p& bmin, const vector3p& bmax )
	{
		min = vector3( bmin.x, bmin.y, bmin.z );
		max = vector3( bmax.x, bmax.y, bmax.z );
	}
	void Extend( const vector3 &p ) { min = min.Min( p ), max = max.Max( p ); }
	void Extend( const aabb &bb ) { min = min.Min( bb.min ); max = max.Max( bb.max ); }
	float Volume() const { const vector3 diff = (max - min); return diff.x * diff.y * diff.z; }
	float Area() const { const vector3 diff = (max - min); return 2.f * (diff.x * diff.y + diff.x * diff.z + diff.y * diff.z); }
	bool Empty() const { const vector3 diff = (max - min); return diff.Min() < 0.f; }; 
	vector3 Centroid() const { return 0.5f * (min + max); }
	const int LongestSide() 
	{
		float s[3] = { max.x - min.x, max.y - min.y, max.z - min.z };
		int retval = 0;
		if (s[1] > s[0]) retval = 1;
		if (s[2] > s[retval]) retval = 2;
		return retval;
	}
	vector3 min, max;
};

class matrix
{
public:
	enum 
	{ 
		TX=3, 
		TY=7, 
		TZ=11, 
		D0=0, D1=5, D2=10, D3=15, 
		SX=D0, SY=D1, SZ=D2, 
		W=D3 
	};
	matrix() { Identity(); }
	float& operator [] ( int a_N ) { return cell[a_N]; }
	void Identity();
	void Rotate( vector3& a_Pos, float a_RX, float a_RY, float a_RZ );
	void RotateX( float a_RX );
	void RotateY( float a_RY );
	void RotateZ( float a_RZ );
	void Translate( vector3& a_Pos ) { cell[TX] += a_Pos.x; cell[TY] += a_Pos.y; cell[TZ] += a_Pos.z; }
	void SetTranslation( vector3& a_Pos ) { cell[TX] = a_Pos.x; cell[TY] = a_Pos.y; cell[TZ] = a_Pos.z; }
	void Normalize();
	void Concatenate( matrix& m2 );
	vector3 Transform( const vector3& v );
	vector3 GetTranslation() { return vector3(cell[TX], cell[TY], cell[TZ]); }
	void Invert();
	float cell[16];
};



//! Quaternion class for representing rotations.
/** It provides cheap combinations and avoids gimbal locks.
Also useful for interpolations. */
class quaternion
{
public:

	//! Default Constructor
	quaternion() : X(0.0f), Y(0.0f), Z(0.0f), W(1.0f) {}

	//! Constructor
	quaternion(float x, float y, float z, float w) : X(x), Y(y), Z(z), W(w) { }

	//! Constructor which converts euler angles (radians) to a quaternion
	quaternion(float x, float y, float z);

	//! Constructor which converts euler angles (radians) to a quaternion
	quaternion(const vector3& vec) { set(vec.x, vec.y, vec.z); }

	//! Constructor which converts a matrix to a quaternion
	quaternion(const matrix& mat);

	//! Equalilty operator
	bool operator==(const quaternion& other) const;

	//! inequality operator
	bool operator!=(const quaternion& other) const;

	//! Assignment operator
	inline quaternion& operator=(const quaternion& other);

	//! Matrix assignment operator
	inline quaternion& operator=(const matrix& other);

	//! Add operator
	quaternion operator+(const quaternion& other) const;

	//! Multiplication operator
	quaternion operator*(const quaternion& other) const;

	//! Multiplication operator with scalar
	quaternion operator*(float s) const;

	//! Multiplication operator with scalar
	quaternion& operator*=(float s);

	//! Multiplication operator
	vector3 operator*(const vector3& v) const;

	//! Multiplication operator
	quaternion& operator*=(const quaternion& other);

	//! Calculates the dot product
	inline float dotProduct(const quaternion& other) const;

	//! Sets new quaternion
	inline quaternion& set(float x, float y, float z, float w);

	//! Sets new quaternion based on euler angles (radians)
	quaternion& set(float x, float y, float z);

	//! Sets new quaternion based on euler angles (radians)
	inline quaternion& set(const vector3& vec);

	//! Sets new quaternion from other quaternion
	inline quaternion& set(const quaternion& quat);

	//! returns if this quaternion equals the other one, taking floating point rounding errors into account
	inline bool equals(const quaternion& other,
		const float tolerance = ROUNDING_ERROR_f32) const;

	//! Normalizes the quaternion
	inline quaternion& normalize();

	//! Creates a matrix from this quaternion
	matrix getMatrix() const;

	//! Creates a matrix from this quaternion
	void getMatrix(matrix &dest, const vector3 &translation) const;

	/*!
	Creates a matrix from this quaternion
	Rotate about a center point
	shortcut for
	quaternion q;
	q.rotationFromTo ( vin[i].Normal, forward );
	q.getMatrixCenter ( lookat, center, newPos );

	matrix m2;
	m2.setInverseTranslation ( center );
	lookat *= m2;

	matrix m3;
	m2.setTranslation ( newPos );
	lookat *= m3;

	*/
	void getMatrixCenter(matrix &dest, const vector3 &center, const vector3 &translation) const;

	//! Creates a matrix from this quaternion
	inline void getMatrix_transposed(matrix &dest) const;

	//! Inverts this quaternion
	quaternion& makeInverse();

	//! Set this quaternion to the result of the interpolation between two quaternions
	quaternion& slerp(quaternion q1, quaternion q2, float interpolate);

	//! Create quaternion from rotation angle and rotation axis.
	/** Axis must be unit length.
	The quaternion representing the rotation is
	q = cos(A/2)+sin(A/2)*(x*i+y*j+z*k).
	\param angle Rotation Angle in radians.
	\param axis Rotation axis. */
	quaternion& fromAngleAxis(float angle, const vector3& axis);

	//! Fills an angle (radians) around an axis (unit vector)
	void toAngleAxis(float &angle, vector3& axis) const;

	//! Output this quaternion to an euler angle (radians)
	void toEuler(vector3& euler) const;

	//! Set quaternion to identity
	quaternion& makeIdentity();

	//! Set quaternion to represent a rotation from one vector to another.
	quaternion& rotationFromTo(const vector3& from, const vector3& to);

	//! Quaternion elements.
	float X; // vectorial (imaginary) part
	float Y;
	float Z;
	float W; // real part
};


bool RayIntersectTriangle(const vector3& org, const vector3& dir,
	const vector3& v0, const vector3& v1, const vector3& v2,
	float* t, float* u, float* v, bool bBackFace = false);

bool SegmentIntersectTriangle(const vector3& start, const vector3& end,
	const vector3& v0, const vector3& v1, const vector3& v2,
	float* t, float* u, float* v, bool bBackFace = false);

