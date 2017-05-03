// Template, major revision 3, beta
// IGAD/NHTV - Jacco Bikker - 2006-2009

// Note:
// This version of the template attempts to setup a rendering surface in system RAM
// and copies it to VRAM using DMA. On recent systems, this yields extreme performance,
// and flips are almost instant. For older systems, there is a fall-back path that
// uses a more conventional approach offered by SDL. If your system uses this, the
// window caption will indicate this. In this case, you may want to tweak the video
// mode setup code for optimal performance.

#include "float.h"
#include "template.h"


#pragma warning (disable : 4273)
#pragma warning (disable : 4616)	// 'warning number '1740' not a valid compiler warning' :)
#pragma warning (disable : 1740)	// 'dllexport/dllimport conflict'





// Constructor which converts euler angles to a quaternion
inline quaternion::quaternion(float x, float y, float z)
{
	set(x, y, z);
}



// Constructor which converts a matrix to a quaternion
inline quaternion::quaternion(const matrix& mat)
{
	(*this) = mat;
}


// equal operator
inline bool quaternion::operator==(const quaternion& other) const
{
	return ((X == other.X) &&
		(Y == other.Y) &&
		(Z == other.Z) &&
		(W == other.W));
}

// inequality operator
inline bool quaternion::operator!=(const quaternion& other) const
{
	return !(*this == other);
}

// assignment operator
inline quaternion& quaternion::operator=(const quaternion& other)
{
	X = other.X;
	Y = other.Y;
	Z = other.Z;
	W = other.W;
	return *this;
}

/*
// matrix assignment operator
inline quaternion& quaternion::operator=(const matrix& m)
{
	const float diag = m(0, 0) + m(1, 1) + m(2, 2) + 1;

	if (diag > 0.0f)
	{
		const float scale = sqrtf(diag) * 2.0f; // get scale from diagonal

		// TODO: speed this up
		X = (m(2, 1) - m(1, 2)) / scale;
		Y = (m(0, 2) - m(2, 0)) / scale;
		Z = (m(1, 0) - m(0, 1)) / scale;
		W = 0.25f * scale;
	}
	else
	{
		if (m(0, 0) > m(1, 1) && m(0, 0) > m(2, 2))
		{
			// 1st element of diag is greatest value
			// find scale according to 1st element, and double it
			const float scale = sqrtf(1.0f + m(0, 0) - m(1, 1) - m(2, 2)) * 2.0f;

			// TODO: speed this up
			X = 0.25f * scale;
			Y = (m(0, 1) + m(1, 0)) / scale;
			Z = (m(2, 0) + m(0, 2)) / scale;
			W = (m(2, 1) - m(1, 2)) / scale;
		}
		else if (m(1, 1) > m(2, 2))
		{
			// 2nd element of diag is greatest value
			// find scale according to 2nd element, and double it
			const float scale = sqrtf(1.0f + m(1, 1) - m(0, 0) - m(2, 2)) * 2.0f;

			// TODO: speed this up
			X = (m(0, 1) + m(1, 0)) / scale;
			Y = 0.25f * scale;
			Z = (m(1, 2) + m(2, 1)) / scale;
			W = (m(0, 2) - m(2, 0)) / scale;
		}
		else
		{
			// 3rd element of diag is greatest value
			// find scale according to 3rd element, and double it
			const float scale = sqrtf(1.0f + m(2, 2) - m(0, 0) - m(1, 1)) * 2.0f;

			// TODO: speed this up
			X = (m(0, 2) + m(2, 0)) / scale;
			Y = (m(1, 2) + m(2, 1)) / scale;
			Z = 0.25f * scale;
			W = (m(1, 0) - m(0, 1)) / scale;
		}
	}

	return Normalize();
}
*/

// multiplication operator
quaternion quaternion::operator*(const quaternion& other) const
{
	quaternion tmp;

	tmp.W = (other.W * W) - (other.X * X) - (other.Y * Y) - (other.Z * Z);
	tmp.X = (other.W * X) + (other.X * W) + (other.Y * Z) - (other.Z * Y);
	tmp.Y = (other.W * Y) + (other.Y * W) + (other.Z * X) - (other.X * Z);
	tmp.Z = (other.W * Z) + (other.Z * W) + (other.X * Y) - (other.Y * X);

	return tmp;
}


// multiplication operator
quaternion quaternion::operator*(float s) const
{
	return quaternion(s*X, s*Y, s*Z, s*W);
}

// multiplication operator
inline quaternion& quaternion::operator*=(float s)
{
	X *= s;
	Y *= s;
	Z *= s;
	W *= s;
	return *this;
}

// multiplication operator
inline quaternion& quaternion::operator*=(const quaternion& other)
{
	return (*this = other * (*this));
}

// add operator
inline quaternion quaternion::operator+(const quaternion& b) const
{
	return quaternion(X + b.X, Y + b.Y, Z + b.Z, W + b.W);
}


// Creates a matrix from this quaternion
inline matrix quaternion::getMatrix() const
{
	matrix m;
	getMatrix_transposed(m);
	return m;
}



//Creates a matrix from this quaternion
/*
inline void quaternion::getMatrix(matrix &dest, const vector3 &center) const
{
	float * m = dest.pointer();

	m[0] = 1.0f - 2.0f*Y*Y - 2.0f*Z*Z;
	m[1] = 2.0f*X*Y + 2.0f*Z*W;
	m[2] = 2.0f*X*Z - 2.0f*Y*W;
	m[3] = 0.0f;

	m[4] = 2.0f*X*Y - 2.0f*Z*W;
	m[5] = 1.0f - 2.0f*X*X - 2.0f*Z*Z;
	m[6] = 2.0f*Z*Y + 2.0f*X*W;
	m[7] = 0.0f;

	m[8] = 2.0f*X*Z + 2.0f*Y*W;
	m[9] = 2.0f*Z*Y - 2.0f*X*W;
	m[10] = 1.0f - 2.0f*X*X - 2.0f*Y*Y;
	m[11] = 0.0f;

	m[12] = center.X;
	m[13] = center.Y;
	m[14] = center.Z;
	m[15] = 1.f;

	//dest.setDefinitelyIdentityMatrix ( matrix::BIT_IS_NOT_IDENTITY );
	dest.setDefinitelyIdentityMatrix(false);
}
*/


/*!
Creates a matrix from this quaternion
Rotate about a center point
shortcut for
quaternion q;
q.rotationFromTo ( vin[i].Normal, forward );
q.getMatrix ( lookat, center );

matrix m2;
m2.setInverseTranslation ( center );
lookat *= m2;
*/
/*
inline void quaternion::getMatrixCenter(matrix &dest,
	const vector3 &center,
	const vector3 &translation) const
{
	float * m = dest.pointer();

	m[0] = 1.0f - 2.0f*Y*Y - 2.0f*Z*Z;
	m[1] = 2.0f*X*Y + 2.0f*Z*W;
	m[2] = 2.0f*X*Z - 2.0f*Y*W;
	m[3] = 0.0f;

	m[4] = 2.0f*X*Y - 2.0f*Z*W;
	m[5] = 1.0f - 2.0f*X*X - 2.0f*Z*Z;
	m[6] = 2.0f*Z*Y + 2.0f*X*W;
	m[7] = 0.0f;

	m[8] = 2.0f*X*Z + 2.0f*Y*W;
	m[9] = 2.0f*Z*Y - 2.0f*X*W;
	m[10] = 1.0f - 2.0f*X*X - 2.0f*Y*Y;
	m[11] = 0.0f;

	dest.setRotationCenter(center, translation);
}
*/
// Creates a matrix from this quaternion
/*
inline void quaternion::getMatrix_transposed(matrix &dest) const
{
	dest[0] = 1.0f - 2.0f*Y*Y - 2.0f*Z*Z;
	dest[4] = 2.0f*X*Y + 2.0f*Z*W;
	dest[8] = 2.0f*X*Z - 2.0f*Y*W;
	dest[12] = 0.0f;

	dest[1] = 2.0f*X*Y - 2.0f*Z*W;
	dest[5] = 1.0f - 2.0f*X*X - 2.0f*Z*Z;
	dest[9] = 2.0f*Z*Y + 2.0f*X*W;
	dest[13] = 0.0f;

	dest[2] = 2.0f*X*Z + 2.0f*Y*W;
	dest[6] = 2.0f*Z*Y - 2.0f*X*W;
	dest[10] = 1.0f - 2.0f*X*X - 2.0f*Y*Y;
	dest[14] = 0.0f;

	dest[3] = 0.f;
	dest[7] = 0.f;
	dest[11] = 0.f;
	dest[15] = 1.f;
	//dest.setDefinitelyIdentityMatrix ( matrix::BIT_IS_NOT_IDENTITY );
	dest.setDefinitelyIdentityMatrix(false);
}
*/


// Inverts this quaternion
inline quaternion& quaternion::makeInverse()
{
	X = -X; Y = -Y; Z = -Z;
	return *this;
}

// sets new quaternion
inline quaternion& quaternion::set(float x, float y, float z, float w)
{
	X = x;
	Y = y;
	Z = z;
	W = w;
	return *this;
}


// sets new quaternion based on euler angles
quaternion& quaternion::set(float x, float y, float z)
{
	float angle;

	angle = x * 0.5;
	const float sr = sin(angle);
	const float cr = cos(angle);

	angle = y * 0.5;
	const float sp = sin(angle);
	const float cp = cos(angle);

	angle = z * 0.5;
	const float sy = sin(angle);
	const float cy = cos(angle);

	const float cpcy = cp * cy;
	const float spcy = sp * cy;
	const float cpsy = cp * sy;
	const float spsy = sp * sy;

	X = (float)(sr * cpcy - cr * spsy);
	Y = (float)(cr * spcy + sr * cpsy);
	Z = (float)(cr * cpsy - sr * spcy);
	W = (float)(cr * cpcy + sr * spsy);

	return normalize();
}

// sets new quaternion based on euler angles
inline quaternion& quaternion::set(const vector3& vec)
{
	return set(vec.x, vec.y, vec.z);
}

// sets new quaternion based on other quaternion
inline quaternion& quaternion::set(const quaternion& quat)
{
	return (*this = quat);
}


//! returns if this quaternion equals the other one, taking floating point rounding errors into account
inline bool quaternion::equals(const quaternion& other, const float tolerance) const
{
	return fequals(X, other.X, tolerance) &&
		fequals(Y, other.Y, tolerance) &&
		fequals(Z, other.Z, tolerance) &&
		fequals(W, other.W, tolerance);
}


// Normalizes the quaternion
inline quaternion& quaternion::normalize()
{
	const float n = X*X + Y*Y + Z*Z + W*W;

	if (n == 1)
		return *this;

	//n = 1.0f / sqrtf(n);
	return (*this *= reciprocal_squareroot(n));
}


// set this quaternion to the result of the interpolation between two quaternions
inline quaternion& quaternion::slerp(quaternion q1, quaternion q2, float time)
{
	float angle = q1.dotProduct(q2);

	if (angle < 0.0f)
	{
		q1 *= -1.0f;
		angle *= -1.0f;
	}

	float scale;
	float invscale;

	if ((angle + 1.0f) > 0.05f)
	{
		if ((1.0f - angle) >= 0.05f) // spherical interpolation
		{
			const float theta = acosf(angle);
			const float invsintheta = reciprocal(sinf(theta));
			scale = sinf(theta * (1.0f - time)) * invsintheta;
			invscale = sinf(theta * time) * invsintheta;
		}
		else // linear interploation
		{
			scale = 1.0f - time;
			invscale = time;
		}
	}
	else
	{
		q2.set(-q1.Y, q1.X, -q1.W, q1.Z);
		scale = sinf(PI * (0.5f - time));
		invscale = sinf(PI * time);
	}

	return (*this = (q1*scale) + (q2*invscale));
}


// calculates the dot product
inline float quaternion::dotProduct(const quaternion& q2) const
{
	return (X * q2.X) + (Y * q2.Y) + (Z * q2.Z) + (W * q2.W);
}


//! axis must be unit length
//! angle in radians
inline quaternion& quaternion::fromAngleAxis(float angle, const vector3& axis)
{
	const float fHalfAngle = 0.5f*angle;
	const float fSin = sinf(fHalfAngle);
	W = cosf(fHalfAngle);
	X = fSin*axis.x;
	Y = fSin*axis.y;
	Z = fSin*axis.z;
	return *this;
}


inline void quaternion::toAngleAxis(float &angle, vector3 &axis) const
{
	const float scale = sqrtf(X*X + Y*Y + Z*Z);

	if (iszero(scale) || W > 1.0f || W < -1.0f)
	{
		angle = 0.0f;
		axis.x = 0.0f;
		axis.y = 1.0f;
		axis.z = 0.0f;
	}
	else
	{
		const float invscale = reciprocal(scale);
		angle = 2.0f * acosf(W);
		axis.x = X * invscale;
		axis.y = Y * invscale;
		axis.z = Z * invscale;
	}
}

inline void quaternion::toEuler(vector3& euler) const
{
	const float sqw = W*W;
	const float sqx = X*X;
	const float sqy = Y*Y;
	const float sqz = Z*Z;

	// heading = rotation about z-axis
	euler.z = (float)(atan2(2.0f * (X*Y + Z*W), (sqx - sqy - sqz + sqw)));

	// bank = rotation about x-axis
	euler.x = (float)(atan2(2.0f * (Y*Z + X*W), (-sqx - sqy + sqz + sqw)));

	// attitude = rotation about y-axis
	euler.y = asinf(clamp(-2.0f * (X*Z - Y*W), -1.0f, 1.0f));
}


vector3 quaternion::operator* (const vector3& v) const
{
	// nVidia SDK implementation

	vector3 uv, uuv;
	vector3 qvec(X, Y, Z);
	uv = qvec.Cross(v);
	uuv = qvec.Cross(uv);
	uv *= (2.0f * W);
	uuv *= 2.0f;

	return v + uv + uuv;
}

// set quaternion to identity
inline quaternion& quaternion::makeIdentity()
{
	W = 1.f;
	X = 0.f;
	Y = 0.f;
	Z = 0.f;
	return *this;
}

quaternion& quaternion::rotationFromTo(const vector3& from, const vector3& to)
{
	// Based on Stan Melax's article in Game Programming Gems
	// Copy, since cannot modify local
	const vector3& v0 = from;
	const vector3& v1 = to;
	//v0.Normalize();
	//v1.Normalize();

	const float d = v0.Dot(v1);
	if (d >= 1.0f) // If dot == 1, vectors are the same
	{
		return makeIdentity();
	}
	else if (d <= -1.0f) // exactly opposite
	{
		vector3 axis(1.0f, 0.f, 0.f);
		axis = axis.Cross( v0 );
		if (axis.SqrLength() < 0.0001f)
		{
			axis.Set(0.f, 1.f, 0.f);
			axis = axis.Cross( v0 );
		}
		return this->fromAngleAxis(PI, axis).normalize();
	}

	const float s = sqrtf((1 + d) * 2); // optimize inv_sqrt
	const float invs = 1.f / s;
	const vector3 c = v0.Cross(v1)*invs;
	X = c.x;
	Y = c.y;
	Z = c.z;
	W = s * 0.5f;

	return *this;
}



void matrix::Identity()
{
	cell[1] = cell[2] = cell[TX] = cell[4] = cell[6] = cell[TY] =
	cell[8] = cell[9] = cell[TZ] = cell[12] = cell[13] = cell[14] = 0;
	cell[D0] = cell[D1] = cell[D2] = cell[W] = 1;
}

void matrix::Rotate( vector3& a_Pos, float a_RX, float a_RY, float a_RZ )
{
	matrix t;
	t.RotateX( a_RZ );
	RotateY( a_RY );
	Concatenate( t );
	t.RotateZ( a_RX );
	Concatenate( t );
	Translate( a_Pos );
}

void matrix::RotateX( float a_RX )
{
	float sx = (float)sin( a_RX * PI / 180 );
	float cx = (float)cos( a_RX * PI / 180 );
	Identity();
	cell[5] = cx, cell[6] = sx, cell[9] = -sx, cell[10] = cx;
}

void matrix::RotateY( float a_RY )
{
	float sy = (float)sin( a_RY * PI / 180 );
	float cy = (float)cos( a_RY * PI / 180 );
	Identity ();
	cell[0] = cy, cell[2] = -sy, cell[8] = sy, cell[10] = cy;
}

void matrix::RotateZ( float a_RZ )
{
	float sz = (float)sin( a_RZ * PI / 180 );
	float cz = (float)cos( a_RZ * PI / 180 );
	Identity ();
	cell[0] = cz, cell[1] = sz, cell[4] = -sz, cell[5] = cz;
}

void matrix::Normalize()
{
	float rclx = 1.0f / sqrtf( cell[0] * cell[0] + cell[1] * cell[1] + cell[2] * cell[2] );
	float rcly = 1.0f / sqrtf( cell[4] * cell[4] + cell[5] * cell[5] + cell[6] * cell[6] );
	float rclz = 1.0f / sqrtf( cell[8] * cell[8] + cell[9] * cell[9] + cell[10] * cell[10] );
	cell[0] *= rclx; cell[1] *= rclx; cell[2] *= rclx;
	cell[4] *= rcly; cell[5] *= rcly; cell[10] *= rcly;
	cell[8] *= rclz; cell[9] *= rclz; cell[11] *= rclz;
}

void matrix::Concatenate( matrix& m2 )
{
	matrix res;
	int c;
	for ( c = 0; c < 4; c++ ) for ( int r = 0; r < 4; r++ )
		res.cell[r * 4 + c] = cell[r * 4] * m2.cell[c] +
			  				  cell[r * 4 + 1] * m2.cell[c + 4] +
							  cell[r * 4 + 2] * m2.cell[c + 8] +
							  cell[r * 4 + 3] * m2.cell[c + 12];
	for ( c = 0; c < 16; c++ ) cell[c] = res.cell[c];
}

vector3 matrix::Transform( const vector3& v )
{
	float x  = cell[0] * v.x + cell[1] * v.y + cell[2] * v.z + cell[3];
	float y  = cell[4] * v.x + cell[5] * v.y + cell[6] * v.z + cell[7];
	float z  = cell[8] * v.x + cell[9] * v.y + cell[10] * v.z + cell[11];
	return vector3( x, y, z );
}

void matrix::Invert()
{
	matrix t;
	int h, i;
	float tx = -cell[3], ty = -cell[7], tz = -cell[11];
	for ( h = 0; h < 3; h++ ) for ( int v = 0; v < 3; v++ ) t.cell[h + v * 4] = cell[v + h * 4];
	for ( i = 0; i < 11; i++ ) cell[i] = t.cell[i];
	cell[3] = tx * cell[0] + ty * cell[1] + tz * cell[2];
	cell[7] = tx * cell[4] + ty * cell[5] + tz * cell[6];
	cell[11] = tx * cell[8] + ty * cell[9] + tz * cell[10];
}


bool RayIntersectTriangle(const vector3& org, const vector3& dir,
	const vector3& v0, const vector3& v1, const vector3& v2,
	float* t, float* u, float* v, bool bBackFace)
{
	// find vectors for two edges sharing v0
	vector3 edge1 = v1 - v0;
	vector3 edge2 = v2 - v0;

	// begin calculating determinant - also used to calculate U parameter
	vector3 pvec = dir.Cross(edge2);

	// calculate distance from v0 to ray origin
	vector3 tvec = org - v0;

	// if determinant is near zero, ray lies in plane of triangle
	float det = edge1.Dot(pvec);

	if (bBackFace && det < 0) { // also test back side
		det = -det;
		tvec = -tvec;
	}

	if (det < 0.0001f) return false;

	// calculate U parameter and test bounds
	*u = tvec.Dot(pvec);
	if (*u < 0.0f || *u > det) return false;

	// prepare to test V parameter
	vector3 qvec = tvec.Cross(edge1);

	// calculate V parameter and test bounds
	*v = dir.Dot(qvec);
	if (*v < 0.0f || *u + *v > det) return false;

	// calculate t, scale parameters, ray intersects triangle
	*t = edge2.Dot(qvec);
	float fInvDet = 1.0f / det;
	*t *= fInvDet;
	*u *= fInvDet;
	*v *= fInvDet;

	return true;
}

bool SegmentIntersectTriangle(const vector3& start, const vector3& end,
	const vector3& v0, const vector3& v1, const vector3& v2,
	float* t, float* u, float* v, bool bBackFace)
{
	if (!RayIntersectTriangle(start, end-start, v0, v1, v2, t, u, v, bBackFace)) return false;
	if (*t < 0 || *t >= 1) return false;
	return true;
	/*
	vector3 edge1 = v1 - v0;
	vector3 edge2 = v2 - v0;
	vector3 normal = edge1.Cross(edge2);
	float fDotStart = normal.Dot(start - v0);
	float fDotEnd = normal.Dot(end - v0);

	float tt = -fDotStart / (fDotEnd - fDotStart);

	if ((fDotStart <= 0.0f && fDotEnd <= 0.0f) || (fDotStart >= 0.0f && fDotEnd >= 0.0f)) return false;

	return true;
	*/
}

