#include "template.h"

extern "C" {
#include "..\client.h"
	extern trace_t		CL_PMTrace_thread(int thread, vec3_t start, vec3_t mins, vec3_t maxs, vec3_t end);
	extern int			r_numdlights;
	extern dlight_t		r_dlights[MAX_DLIGHTS];
	extern qboolean		gBlockOnly[MAX_TRACE_THREAD_NUM];

}

cvar_t* tl_test = NULL; // for teset

cvar_t* tl_interlaced = NULL; // line interlace
cvar_t* tl_exposure = NULL; // exposure factor
cvar_t* tl_aorange = NULL; // testing range for ao lighting
cvar_t* tl_dlightsnum = NULL; // max dynamic lights number
cvar_t* tl_dlights_shadow_num = NULL; // Up to how many lights can have shadow?


//#define SINGLE_THREAD_RENDERING	// whether use single thread rendering

static HANDLE thread_handles[MAX_TRACE_THREAD_NUM];



typedef struct { int x; int y; } iv2d;

#define V3T( v )	( *(vec3_t*)(float*)v )

#define dmRand() (((float)rand()) / RAND_MAX)



static const int W = 128;
static const int H = 128;
float gNearClip = H/2;


static const int gNormalNum = 8 * 2;
static vector3 vRadomNormal[gNormalNum];

static void GenerateRadomNormals()
{
	//static bool bInit = false;
	//if (bInit) return;
	//bInit = true;
	const float vrange = 0.5f;
	int i = 0;
	while (true) {
		float fx = dmRand() * 2 * vrange - vrange;
		float fy = dmRand() * 2 * vrange - vrange;
		if (fx*fx + fy*fy >= 1) continue;

		float fz = sqrt(1.0f - fx*fx - fy*fy);
		vRadomNormal[i++] = vector3(fx, fy, fz);
		if (i >= gNormalNum) break;
	}

}

float Expose(float light, float exposure)
{
	return (1.0f - exp(-light * exposure));
}

unsigned pixelcolor(int thread, int x, int y)
	{
		int r = 0, g = 0, b = 0, a = 255;


		vector3 vieworg(cl.refdef.vieworg);

		vector3 right(cl.v_right);
		vector3 up(cl.v_up);
		vector3 forward(cl.v_forward);

		vector3 worldx(right.x, up.x, forward.x);
		vector3 worldy(right.y, up.y, forward.y);
		vector3 worldz(right.z, up.z, forward.z);

		#define RAYLEN 1500

		vector3 rdv(x, 0.75f*y, gNearClip); // view space
		rdv.Normalize();
		vector3 raydir(rdv.Dot(worldx), rdv.Dot(worldy), rdv.Dot(worldz)); // world space
		vector3 end = vieworg + raydir * RAYLEN;


		trace_t t = CL_PMTrace_thread(thread, V3T(vieworg), vec3_origin, vec3_origin, V3T(end));
		if (1 == t.fraction) { // don't hit anything
			r = 255, g = 150, b = 100;
			goto output;
		}
		if (t.surface && t.surface->flags & SURF_SKY) {
			r = 255, g = 150, b = 100;
			goto output;
		}


		//r=g=b = t.fraction * 128;
		vector3 pnormal(t.plane.normal);
		quaternion rot;
		rot.rotationFromTo( vector3(0,0,1), pnormal );

		vector3 start = t.endpos;
		vector3 pn;
		float sum = 0;
		float skylight = 0;

		// ao lighting
		if ( tl_aorange->value > 0) {
			
			GenerateRadomNormals();

			for (int r = 0; r < gNormalNum; r++) {
				pn = rot * vRadomNormal[r];
				end = start + pn / pn.Length() * tl_aorange->value;
				trace_t tt = CL_PMTrace_thread(thread, V3T(start), vec3_origin, vec3_origin, V3T(end));
				if (tt.surface) {
					if (tt.surface->flags & SURF_SKY) skylight += 0.1f;
				}
				sum += tt.fraction;
			}

			float exposure = tl_exposure->value * tl_aorange->value / gNormalNum;
			r = g = b = Expose(sum, exposure) * 255;
			skylight = Expose(skylight, exposure);
			r += skylight * 255;
			g += skylight * 80;
		}

		// dynamic lights
		int dlightsnum = min(tl_dlightsnum->value, r_numdlights);
		int shadowcount = 0;
		for (int i = 0; i < dlightsnum; i++) {
			const dlight_t& dl = r_dlights[i];
			if (dl.intensity <= 0) continue;

			vector3 lightpos(dl.origin);

			if (lightpos.Dot(pnormal) - t.plane.dist <= 0) continue; // light in back of plane

			vector3 tolight = lightpos - start;
			float dist = tolight.Length();

			if (dist > dl.range) continue;

			tolight = tolight/dist; // normzize
			float diffuse = pnormal.Dot(tolight);
			float atten = 0.1f / (1.0f + 0.5f * dist + 0.005f * dist*dist );
			diffuse *= atten * dl.intensity;
			if ( diffuse > 0.05f) {
				if (shadowcount++ < tl_dlights_shadow_num->value) {
					gBlockOnly[thread] = TRUE; // faster tracing, no need distance
					trace_t tt = CL_PMTrace_thread(thread, V3T(start), vec3_origin, vec3_origin, V3T(lightpos));
					gBlockOnly[thread] = FALSE;
					if (tt.fraction < 1.0f) { // hit something
						diffuse = (1 - diffuse)*0.053f;
					}
				}
			}
			r += diffuse * dl.color[0] * 255;
			g += diffuse * dl.color[1] * 255;
			b += diffuse * dl.color[2] * 255;
		}



		if (r > 255) r = 255;
		if (g > 255) g = 255;
		if (b > 255) b = 255;



output:

		return a << 24 | b << 16 | g << 8 | r;
	}


	class CTile
	{
	public:
		static const int pixelnum = 64 * 64;
	private:
		iv2d pixels[pixelnum];
		int count;
	public:
		CTile() {
			clear();
		}
		void clear() { count = 0;  }
		inline const iv2d& getPixel(int i) { return pixels[i]; }
		bool pushpixel(const iv2d& iv) // return true if tile is full
		{
			pixels[count++] = iv;
			if (count >= pixelnum) return true;
			return false;
		}
		
	};

static CTile	tiles[MAX_TRACE_THREAD_NUM];
static int	tilenum = 0;


static unsigned pic[W*H];
static const int halfW = W / 2;
static const int halfH = H / 2;


unsigned int tile_thread_proc(void* param)
{
	int t = (int)param;

	for (int p = 0; p < CTile::pixelnum; p++) {
		int w = tiles[t].getPixel(p).x;
		int h = tiles[t].getPixel(p).y;
		int i = h*W + w;
		pic[i] = pixelcolor( t, w - halfW, H - h - halfH);
	}

	return 0;
}


extern "C" {



	void PathTracingFrame()
	{
		if (!tl_interlaced) {
			tl_test = Cvar_Get("tl_test", "999", 0);
			tl_interlaced = Cvar_Get("tl_interlaced", "0", 0);
			tl_exposure = Cvar_Get("tl_exposure", "0.001", 0);
			tl_aorange = Cvar_Get("tl_aorange", "600", 0);
			tl_dlightsnum = Cvar_Get("tl_dlightsnum", "32", 0);
			tl_dlights_shadow_num = Cvar_Get("tl_dlights_shadow_num", "8", 0);
		}

		static iv2d pc[4] = {
			{ 0, 0 }, { 1, 1 }, { 0, 1 }, {1,0}
		};
		static int pci = 0;

		const iv2d& pp = pc[pci++];
		if (pci >= 4) pci = 0;

		static int odd = 0;
		odd = !odd;


		iv2d iv;

		// first clear tiles
		for (int t = 0; t < MAX_TRACE_THREAD_NUM; t++ ) tiles[t].clear();
		tilenum = 0;

		// push pixels to tiles
		for (int h = 0; h < H; h++)
			for (int w = 0; w < W; w++) {
				//if (w % 2 == pp.x && h % 2 == pp.y) {
				if (!tl_interlaced->value || h % 2 == odd) {
					iv.x = w; iv.y = h;
					if (tiles[tilenum].pushpixel(iv))
						tilenum++;
				}
			}

		// render tiles
		for (int t = 0; t < tilenum; t++) {
			#ifdef SINGLE_THREAD_RENDERING
				tile_thread_proc( (void*)t );
			#else
				DWORD threadId = 0;
				thread_handles[t] = (HANDLE)CreateThread(NULL, 0, 
					(LPTHREAD_START_ROUTINE)tile_thread_proc, (void*)t, 0, &threadId);
			#endif
		}

#ifndef SINGLE_THREAD_RENDERING
		WaitForMultipleObjects(tilenum, thread_handles, true, INFINITE);
#endif

		re.DrawStretchRaw(0, 0, viddef.width, viddef.height,
		//re.DrawStretchRaw(0, 0, 256,256,
			W, H, (byte*)pic, true);
	}

}

