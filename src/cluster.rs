//! **Experimental** meshlet building and cluster bounds generation

use crate::quantize::quantize_snorm;
use crate::util::zero_inverse;
use crate::vertex::Position;

use std::convert::TryInto;

/// Bounds returned by `compute_cluster/meshlet_bounds`.
///
/// `cone_axis_s8` and `cone_cutoff_s8` are stored in 8-bit SNORM format; decode them using `x/127.0`.
///
/// * Bounding sphere: useful for backface culling
/// * Normal cone: useful for backface culling
#[derive(Default)]
pub struct Bounds {
	/// Bounding sphere center
	pub center: [f32; 3],
	/// Bounding sphere radius
	pub radius: f32,
	/// Normal cone apex
	pub cone_apex: [f32; 3],
	/// Normal cone axis
	pub cone_axis: [f32; 3],
	/// Normal cone cutoff
	///
	/// Can be calculated from angle using `cos(angle/2)`.
	pub cone_cutoff: f32,
	/// Normal cone axis
	pub cone_axis_s8: [i8; 3],
	/// Normal cone cutoff
	pub cone_cutoff_s8: i8,
}

#[derive(Clone)]
pub struct Meshlet {
    pub vertices: [i32; 64],
	pub indices: [[u8; 3]; 126],
	pub triangle_count: u8,
	pub vertex_count: u8,
}

impl Meshlet {
    pub const VERTICES_COUNT: usize = 64;
    pub const TRIANGLES_COUNT: usize = 126;
}

impl Default for Meshlet {
    fn default() -> Self {
        Meshlet {
            vertices: [0; Self::VERTICES_COUNT],
            indices: [Default::default(); Self::TRIANGLES_COUNT],
			triangle_count: Default::default(),
			vertex_count: Default::default(),
        }
    }
}

fn compute_bounding_sphere(points: &[[f32; 3]]) -> [f32; 4] {
    assert!(!points.is_empty());
    
	// find extremum points along all 3 axes; for each axis we get a pair of points with min/max coordinates
	let mut pmin = [[f32::MAX; 3]; 3];
	let mut pmax = [[f32::MIN; 3]; 3];

	for p in points {
		for axis in 0..3 {
			if p[axis] < pmin[axis][axis] { pmin[axis] = *p; }
			if p[axis] > pmax[axis][axis] { pmax[axis] = *p; }
		}
	}

	// find the pair of points with largest distance
	let mut paxisd2 = 0.0;
	let mut paxis = 0;

	for axis in 0..3 {
		let p1 = pmin[axis];
		let p2 = pmax[axis];

		let d2 = (p2[0] - p1[0]) * (p2[0] - p1[0]) + (p2[1] - p1[1]) * (p2[1] - p1[1]) + (p2[2] - p1[2]) * (p2[2] - p1[2]);

		if d2 > paxisd2 {
			paxisd2 = d2;
			paxis = axis;
		}
	}

	// use the longest segment as the initial sphere diameter
	let p1 = pmin[paxis];
	let p2 = pmax[paxis];

	let mut center: [f32; 3] =[
        (p1[0] + p2[0]) / 2.0, 
        (p1[1] + p2[1]) / 2.0, 
        (p1[2] + p2[2]) / 2.0,
    ];
	let mut radius = paxisd2.sqrt() / 2.0;

	// iteratively adjust the sphere up until all points fit
	for p in points {
		let d2 = (p[0] - center[0]) * (p[0] - center[0]) + (p[1] - center[1]) * (p[1] - center[1]) + (p[2] - center[2]) * (p[2] - center[2]);

		if d2 > radius * radius {
			let d = d2.sqrt();
			assert!(d > 0.0);

			let k = 0.5 + (radius / d) / 2.0;

			center[0] = center[0] * k + p[0] * (1.0 - k);
			center[1] = center[1] * k + p[1] * (1.0 - k);
			center[2] = center[2] * k + p[2] * (1.0 - k);
			radius = (radius + d) / 2.0;
		}
	}

    [
        center[0],
        center[1],
        center[2],
        radius,
    ]
}

/// Returns worst case size requirement for [build_meshlets].
pub fn build_meshlets_bound(index_count: usize, max_vertices: usize, max_triangles: usize) -> usize {
	assert!(index_count % 3 == 0);
	assert!(max_vertices >= 3);
	assert!(max_triangles >= 1);

	// meshlet construction is limited by max vertices and max triangles per meshlet
	// the worst case is that the input is an unindexed stream since this equally stresses both limits
	// note that we assume that in the worst case, we leave 2 vertices unpacked in each meshlet - if we have space for 3 we can pack any triangle
	let max_vertices_conservative = max_vertices - 2;
	let meshlet_limit_vertices = (index_count + max_vertices_conservative - 1) / max_vertices_conservative;
	let meshlet_limit_triangles = (index_count / 3 + max_triangles - 1) / max_triangles;

	if meshlet_limit_vertices > meshlet_limit_triangles {
        meshlet_limit_vertices
    } else {
        meshlet_limit_triangles
    }
}

/// Splits the mesh into a set of meshlets where each meshlet has a micro index buffer indexing into meshlet vertices that refer to the original vertex buffer.
///
/// The resulting data can be used to render meshes using NVidia programmable mesh shading pipeline, or in other cluster-based renderers.
/// For maximum efficiency the index buffer being converted has to be optimized for vertex cache first.
///
/// # Arguments
///
/// * `destination`: must contain enough space for all meshlets, worst case size can be computed with [build_meshlets_bound]
/// * `max_vertices` and `max_triangles`: can't exceed limits statically declared in [Meshlet] (`VERTICES_COUNT` and `TRIANGLES_COUNT`)
pub fn build_meshlets(destination: &mut [Meshlet], indices: &[u32], vertex_count: usize, max_vertices: usize, max_triangles: usize) -> usize {
	assert!(indices.len() % 3 == 0);
	assert!(max_vertices >= 3);
	assert!(max_triangles >= 1);

	let mut meshlet = Meshlet::default();

	assert!(max_vertices <= Meshlet::VERTICES_COUNT);
	assert!(max_triangles <= Meshlet::TRIANGLES_COUNT);

	const UNUSED: u8 = 0xff;

    // index of the vertex in the meshlet, `UNUSED` if the vertex isn't used
    let mut used = vec![UNUSED; vertex_count];

	let mut offset = 0;

	for i in (0..indices.len()).step_by(3) {
        let a = indices[i + 0] as usize;
        let b = indices[i + 1] as usize;
        let c = indices[i + 2] as usize;

		assert!(a < vertex_count && b < vertex_count && c < vertex_count);

		let used_extra = [used[a], used[b], used[c]].iter().filter(|v| **v == UNUSED).count();

		if meshlet.vertex_count as usize + used_extra > max_vertices || meshlet.triangle_count as usize >= max_triangles {
            destination[offset] = meshlet.clone();
            offset += 1;

			for j in 0..meshlet.vertex_count {
                used[meshlet.vertices[j as usize] as usize] = UNUSED;
            }

            meshlet = Meshlet::default();
		}

		if used[a] == UNUSED {
			used[a] = meshlet.vertex_count;
            meshlet.vertices[meshlet.vertex_count as usize] = a.try_into().unwrap();
            meshlet.vertex_count += 1;
		}

		if used[b] == UNUSED {
			used[b] = meshlet.vertex_count;
            meshlet.vertices[meshlet.vertex_count as usize] = b.try_into().unwrap();
            meshlet.vertex_count += 1;
		}

		if used[c] == UNUSED {
			used[c] = meshlet.vertex_count;
            meshlet.vertices[meshlet.vertex_count as usize] = c.try_into().unwrap();
            meshlet.vertex_count += 1;
		}

		meshlet.indices[meshlet.triangle_count as usize][..].copy_from_slice(&[used[a], used[b], used[c]]);

		meshlet.triangle_count += 1;
	}

	if meshlet.triangle_count > 0 {
        destination[offset] = meshlet;
        offset += 1;
    }

	assert!(offset <= build_meshlets_bound(indices.len(), max_vertices, max_triangles));

	offset
}

/// Creates bounding volumes that can be used for frustum, backface and occlusion culling.
///
/// For backface culling with orthographic projection, use the following formula to reject backfacing clusters:
/// ```glsl
/// dot(view, cone_axis) >= cone_cutoff
/// ```
///
/// For perspective projection, you can the formula that needs cone apex in addition to axis & cutoff:
/// ```glsl
/// dot(normalize(cone_apex - camera_position), cone_axis) >= cone_cutoff
/// ```
///
/// Alternatively, you can use the formula that doesn't need cone apex and uses bounding sphere instead:
/// ```glsl
/// dot(normalize(center - camera_position), cone_axis) >= cone_cutoff + radius / length(center - camera_position)
/// ```
/// or an equivalent formula that doesn't have a singularity at center = camera_position:
/// ```glsl
/// dot(center - camera_position, cone_axis) >= cone_cutoff * length(center - camera_position) + radius
/// ```
///
/// The formula that uses the apex is slightly more accurate but needs the apex; if you are already using bounding sphere
/// to do frustum/occlusion culling, the formula that doesn't use the apex may be preferable.
///
/// # Arguments
///
/// * `indices`: should be smaller than or equal to 256*3 (the function assumes clusters of limited size)
pub fn compute_cluster_bounds<Vertex>(indices: &[u32], vertices: &[Vertex]) -> Bounds
where
	Vertex: Position
{
	assert!(indices.len() % 3 == 0);
	assert!(indices.len() / 3 <= 256);

	// compute triangle normals and gather triangle corners
	let mut normals: [[f32; 3]; 256] = [Default::default(); 256];
	let mut corners: [[[f32; 3]; 3]; 256] = [Default::default(); 256];
	let mut triangles = 0;

	let vertex_count = vertices.len();

	for i in (0..indices.len()).step_by(3) {
        let a = indices[i + 0] as usize;
        let b = indices[i + 1] as usize;
        let c = indices[i + 2] as usize;
        assert!(a < vertex_count && b < vertex_count && c < vertex_count);

		let p0 = vertices[a].pos();
		let p1 = vertices[b].pos();
		let p2 = vertices[c].pos();

		let p10 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
		let p20 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];

		let normalx = p10[1] * p20[2] - p10[2] * p20[1];
		let normaly = p10[2] * p20[0] - p10[0] * p20[2];
		let normalz = p10[0] * p20[1] - p10[1] * p20[0];

		let area = (normalx * normalx + normaly * normaly + normalz * normalz).sqrt();

		// no need to include degenerate triangles - they will be invisible anyway
		if area == 0.0 {
            continue;
        }

		// record triangle normals & corners for future use; normal and corner 0 define a plane equation
		normals[triangles][0] = normalx / area;
		normals[triangles][1] = normaly / area;
        normals[triangles][2] = normalz / area;

        corners[triangles][0].copy_from_slice(&p0);
        corners[triangles][1].copy_from_slice(&p1);
        corners[triangles][2].copy_from_slice(&p2);

		triangles += 1;
	}

	let mut bounds = Bounds::default();

	// degenerate cluster, no valid triangles => trivial reject (cone data is 0)
	if triangles == 0 {
        return bounds;
    }

	// compute cluster bounding sphere; we'll use the center to determine normal cone apex as well
	let psphere = compute_bounding_sphere(unsafe { 
		let x: &[[f32; 3]] = std::mem::transmute(&corners[0..triangles]);
		std::slice::from_raw_parts(x.as_ptr(), triangles * 3)
	});

	let center = [psphere[0], psphere[1], psphere[2]];

	// treating triangle normals as points, find the bounding sphere - the sphere center determines the optimal cone axis
	let nsphere = compute_bounding_sphere(&normals[0..triangles]);

	let mut axis = [nsphere[0], nsphere[1], nsphere[2]];
	let axislength = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
	let invaxislength = zero_inverse(axislength);

	axis[0] *= invaxislength;
	axis[1] *= invaxislength;
	axis[2] *= invaxislength;

	// compute a tight cone around all normals, mindp = cos(angle/2)
	let mut mindp = 1.0;

	for normal in &normals[0..triangles] {
		let dp = normal[0] * axis[0] + normal[1] * axis[1] + normal[2] * axis[2];

		mindp = dp.min(mindp);
	}

	// fill bounding sphere info; note that below we can return bounds without cone information for degenerate cones
	bounds.center = center;
	bounds.radius = psphere[3];

	// degenerate cluster, normal cone is larger than a hemisphere => trivial accept
	// note that if mindp is positive but close to 0, the triangle intersection code below gets less stable
	// we arbitrarily decide that if a normal cone is ~168 degrees wide or more, the cone isn't useful
	if mindp <= 0.1 {
		bounds.cone_cutoff = 1.0;
		bounds.cone_cutoff_s8 = 127;
		return bounds;
	}

	let mut maxt = 0.0;

	// we need to find the point on center-t*axis ray that lies in negative half-space of all triangles
	for i in 0..triangles {
		// dot(center-t*axis-corner, trinormal) = 0
		// dot(center-corner, trinormal) - t * dot(axis, trinormal) = 0
		let corner = corners[i][0];
		let cx = center[0] - corner[0];
		let cy = center[1] - corner[1];
		let cz = center[2] - corner[2];

		let normal = normals[i];
		let dc = cx * normal[0] + cy * normal[1] + cz * normal[2];
		let dn = axis[0] * normal[0] + axis[1] * normal[1] + axis[2] * normal[2];

		// dn should be larger than mindp cutoff above
		assert!(dn > 0.0);
		let t = dc / dn;

		maxt = t.max(maxt);
	}

	// cone apex should be in the negative half-space of all cluster triangles by construction
	bounds.cone_apex[0] = center[0] - axis[0] * maxt;
	bounds.cone_apex[1] = center[1] - axis[1] * maxt;
	bounds.cone_apex[2] = center[2] - axis[2] * maxt;

	// note: this axis is the axis of the normal cone, but our test for perspective camera effectively negates the axis
	bounds.cone_axis[0] = axis[0];
	bounds.cone_axis[1] = axis[1];
	bounds.cone_axis[2] = axis[2];

	// cos(a) for normal cone is mindp; we need to add 90 degrees on both sides and invert the cone
	// which gives us -cos(a+90) = -(-sin(a)) = sin(a) = sqrt(1 - cos^2(a))
	bounds.cone_cutoff = (1.0 - mindp * mindp).sqrt();

	// quantize axis & cutoff to 8-bit SNORM format
	bounds.cone_axis_s8[0] = quantize_snorm(bounds.cone_axis[0], 8).try_into().unwrap();
	bounds.cone_axis_s8[1] = quantize_snorm(bounds.cone_axis[1], 8).try_into().unwrap();
	bounds.cone_axis_s8[2] = quantize_snorm(bounds.cone_axis[2], 8).try_into().unwrap();

	// for the 8-bit test to be conservative, we need to adjust the cutoff by measuring the max. error
	let cone_axis_s8_e0 = (bounds.cone_axis_s8[0] as f32 / 127.0 - bounds.cone_axis[0]).abs();
	let cone_axis_s8_e1 = (bounds.cone_axis_s8[1] as f32 / 127.0 - bounds.cone_axis[1]).abs();
	let cone_axis_s8_e2 = (bounds.cone_axis_s8[2] as f32 / 127.0 - bounds.cone_axis[2]).abs();

	// note that we need to round this up instead of rounding to nearest, hence +1
	let cone_cutoff_s8 = (127.0 * (bounds.cone_cutoff + cone_axis_s8_e0 + cone_axis_s8_e1 + cone_axis_s8_e2) + 1.0) as i32;

	bounds.cone_cutoff_s8 = if cone_cutoff_s8 > 127 {
        127
    } else {
        cone_cutoff_s8.try_into().unwrap()
    };

	bounds
}

/// Creates bounding volumes that can be used for frustum, backface and occlusion culling.
///
/// Same as [compute_cluster_bounds] but with meshlets as input.
pub fn compute_meshlet_bounds<Vertex>(meshlet: &Meshlet, vertices: &[Vertex]) -> Bounds 
where
	Vertex: Position
{
	let mut indices = [0; Meshlet::TRIANGLES_COUNT * 3];

	for i in 0..meshlet.triangle_count as usize {
		let triangle = meshlet.indices[i];

		let a = meshlet.vertices[triangle[0] as usize] as u32;
		let b = meshlet.vertices[triangle[1] as usize] as u32;
		let c = meshlet.vertices[triangle[2] as usize] as u32;

		// note: `compute_cluster_bounds` checks later if a/b/c are in range, no need to do it here

		indices[i*3..i*3+3].copy_from_slice(&[a, b, c]);
	}

	compute_cluster_bounds(&indices[0..meshlet.triangle_count as usize * 3], vertices)
}

#[cfg(test)]
mod test {
	use super::*;
	
    struct Vertex {
        x: f32,
        y: f32,
        z: f32,
    }

	impl Position for Vertex {
		fn pos(&self) -> [f32; 3] {
			[self.x, self.y, self.z]
		}
	}
	
	fn vb_from_slice(slice: &[f32]) -> Vec<Vertex> {
        slice.chunks_exact(3).map(|v| Vertex { x: v[0], y: v[1], z: v[2] }).collect()
    }

    #[test]
    fn test_cluster_bounds_degenerate() {
		let vbd = vb_from_slice(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let ibd = [0, 0, 0];
        let ib1 = [0, 1, 2];
    
        // all of the bounds below are degenerate as they use 0 triangles, one topology-degenerate triangle and one position-degenerate triangle respectively
        let bounds0 = compute_cluster_bounds::<Vertex>(&[], &[]);
        let boundsd = compute_cluster_bounds(&ibd, &vbd);
        let bounds1 = compute_cluster_bounds(&ib1, &vbd);
    
        assert!(bounds0.center[0] == 0.0 && bounds0.center[1] == 0.0 && bounds0.center[2] == 0.0 && bounds0.radius == 0.0);
        assert!(boundsd.center[0] == 0.0 && boundsd.center[1] == 0.0 && boundsd.center[2] == 0.0 && boundsd.radius == 0.0);
        assert!(bounds1.center[0] == 0.0 && bounds1.center[1] == 0.0 && bounds1.center[2] == 0.0 && bounds1.radius == 0.0);
    
        let vb1 = vb_from_slice(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let ib2 = [0, 1, 2, 0, 2, 1];
    
        // these bounds have a degenerate cone since the cluster has two triangles with opposite normals
        let bounds2 = compute_cluster_bounds(&ib2, &vb1);
    
        assert!(bounds2.cone_apex[0] == 0.0 && bounds2.cone_apex[1] == 0.0 && bounds2.cone_apex[2] == 0.0);
        assert!(bounds2.cone_axis[0] == 0.0 && bounds2.cone_axis[1] == 0.0 && bounds2.cone_axis[2] == 0.0);
        assert!(bounds2.cone_cutoff == 1.0);
        assert!(bounds2.cone_axis_s8[0] == 0 && bounds2.cone_axis_s8[1] == 0 && bounds2.cone_axis_s8[2] == 0);
        assert!(bounds2.cone_cutoff_s8 == 127);
    
        // however, the bounding sphere needs to be in tact (here we only check bbox for simplicity)
        assert!(bounds2.center[0] - bounds2.radius <= 0.0 && bounds2.center[0] + bounds2.radius >= 1.0);
        assert!(bounds2.center[1] - bounds2.radius <= 0.0 && bounds2.center[1] + bounds2.radius >= 1.0);
        assert!(bounds2.center[2] - bounds2.radius <= 0.0 && bounds2.center[2] + bounds2.radius >= 1.0);
    
    }
}
