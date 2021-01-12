//! Mesh triangle list ↔ triangle strip conversion

fn find_strip_first(buffer: &[[u32; 3]], valence: &[u32]) -> usize {
	let mut index = 0;
	let mut iv = u32::MAX;

	for (i, b) in buffer.iter().enumerate() {
        let va = valence[b[0] as usize];
        let vb = valence[b[1] as usize];
        let vc = valence[b[2] as usize];

        let v = va.min(vb).min(vc);

		if v < iv {
			index = i;
			iv = v;
		}
	}

	index
}

fn find_strip_next(buffer: &[[u32; 3]], e: (u32, u32)) -> i32 {
	for (i, abc) in buffer.iter().enumerate() {
		if e.0 == abc[0] && e.1 == abc[1] {
			return ((i as i32) << 2) | 2;
        } else if e.0 == abc[1] && e.1 == abc[2] {
			return ((i as i32) << 2) | 0;
        } else if e.0 == abc[2] && e.1 == abc[0] {
            return ((i as i32) << 2) | 1;
        }
	}

	-1
}

/// Converts a previously vertex cache optimized triangle list to triangle strip, stitching strips using restart index or degenerate triangles.
///
/// Returns the number of indices in the resulting strip, with destination containing new index data.
/// For maximum efficiency the index buffer being converted has to be optimized for vertex cache first.
/// Using restart indices can result in ~10% smaller index buffers, but on some GPUs restart indices may result in decreased performance.
///
/// # Arguments
///
/// * `destination`: must contain enough space for the target index buffer, worst case can be computed with [stripify_bound]
/// * `restart_index`: should be `u32::MAX` or `0` to use degenerate triangles
pub fn stripify(destination: &mut [u32], indices: &[u32], vertex_count: usize, restart_index: u32) -> usize {
	assert!(indices.len() % 3 == 0);

	const BUFFER_CAPACITY: usize = 8;

	let mut buffer = [[0; 3]; BUFFER_CAPACITY];
	let mut buffer_size = 0;

	let mut index_offset = 0;

	let mut strip = [0; 2];
	let mut parity = false;

	let mut strip_size = 0;

	// compute vertex valence; this is used to prioritize starting triangle for strips
	let mut valence = vec![0; vertex_count];

	for index in indices {
		valence[*index as usize] += 1;
	}

	let mut next: i32 = -1;

	let index_count = indices.len();

	while buffer_size > 0 || index_offset < index_count {
		assert!(next < 0 || (((next >> 2) as usize) < buffer_size && (next & 3) < 3));

		// fill triangle buffer
		while buffer_size < BUFFER_CAPACITY && index_offset < index_count {
			&buffer[buffer_size].copy_from_slice(&indices[index_offset..index_offset+3]);

			buffer_size += 1;
			index_offset += 3;
		}

		assert!(buffer_size > 0 && buffer_size <= buffer.len());

		if next >= 0 {
			let i = (next >> 2) as usize;
            let a = buffer[i][0] as usize;
            let b = buffer[i][1] as usize;
            let c = buffer[i][2] as usize;
			let v = buffer[i][(next & 3) as usize];

            // ordered removal from the buffer
            let buffer_length = buffer.len();
            buffer.copy_within(i+1..buffer_length, i);
			buffer_size -= 1;

			// update vertex valences for strip start heuristic
			valence[a] -= 1;
			valence[b] -= 1;
			valence[c] -= 1;

			// find next triangle (note that edge order flips on every iteration)
			// in some cases we need to perform a swap to pick a different outgoing triangle edge
            // for [a b c], the default strip edge is [b c], but we might want to use [a c]
			let cont = find_strip_next(&buffer[0..buffer_size], if parity { (strip[1], v) } else { (v, strip[1]) });
			let swap = if cont < 0 {
                find_strip_next(&buffer[0..buffer_size], if parity { (v, strip[0]) } else { (strip[0], v) })
            } else { 
                -1
            };

			if cont < 0 && swap >= 0 {
				// [a b c] => [a b a c]
                destination[strip_size] = strip[0];
                strip_size += 1;
                destination[strip_size] = v;
                strip_size += 1;

				// next strip has same winding
				// ? a b => b a v
				strip[1] = v;

				next = swap;
			} else {
				// emit the next vertex in the strip
                destination[strip_size] = v;
                strip_size += 1;

				// next strip has flipped winding
				strip[0] = strip[1];
				strip[1] = v;
				parity ^= true;

				next = cont;
			}
		} else {
			// if we didn't find anything, we need to find the next new triangle
			// we use a heuristic to maximize the strip length
			let i = find_strip_first(&buffer[0..buffer_size], &valence);
            let mut abc = buffer[i];

            // ordered removal from the buffer
            buffer.copy_within(i+1.., i);
			buffer_size -= 1;

			// update vertex valences for strip start heuristic
			valence[abc[0] as usize] -= 1;
			valence[abc[1] as usize] -= 1;
			valence[abc[2] as usize] -= 1;

			// we need to pre-rotate the triangle so that we will find a match in the existing buffer on the next iteration
			let ea = find_strip_next(&buffer[0..buffer_size], (abc[2], abc[1]));
			let eb = find_strip_next(&buffer[0..buffer_size], (abc[0], abc[2]));
			let ec = find_strip_next(&buffer[0..buffer_size], (abc[1], abc[0]));

			// in some cases we can have several matching edges; since we can pick any edge, we pick the one with the smallest
			// triangle index in the buffer. this reduces the effect of stripification on ACMR and additionally - for unclear
			// reasons - slightly improves the stripification efficiency
			let mut mine = i32::MAX;
			mine = if ea >= 0 && mine > ea { ea } else { mine };
			mine = if eb >= 0 && mine > eb { eb } else { mine };
			mine = if ec >= 0 && mine > ec { ec } else { mine };

			match mine {
                _ if mine == ea => {
                    // keep abc
                    next = ea;
                }
                _ if mine == eb => {
                    // abc -> bca
                    abc.rotate_left(1);
                    next = eb;
                }
                _ if mine == ec => {
                    // abc -> cab
                    abc.rotate_right(1);
                    next = ec;
                }
                _ => {}
            }

			if restart_index != 0 {
				if strip_size != 0 {
                    destination[strip_size] = restart_index;
                    strip_size += 1;
                }

                &destination[strip_size..strip_size+3].copy_from_slice(&abc);
                strip_size += 3;

				// new strip always starts with the same edge winding
				strip[0] = abc[1];
				strip[1] = abc[2];
				parity = true;
			} else {
				if strip_size != 0 {
					// connect last strip using degenerate triangles
                    destination[strip_size] = strip[1];
                    strip_size += 1;
                    destination[strip_size] = abc[0];
                    strip_size += 1;
				}

				// note that we may need to flip the emitted triangle based on parity
				// we always end up with outgoing edge "cb" in the end
				let (e0, e1) = if parity { 
					(abc[2], abc[1])
				} else { 
					(abc[1], abc[2]) 
				};
                
                &destination[strip_size..strip_size+3].copy_from_slice(&[abc[0], e0, e1]);
                strip_size += 3;

				strip[0] = e0;
				strip[1] = e1;
				parity ^= true;
			}
		}
	}

	strip_size
}

/// Returns worst case size requirement for [stripify].
pub fn stripify_bound(index_count: usize) -> usize {
	assert!(index_count % 3 == 0);

	// worst case without restarts is 2 degenerate indices and 3 indices per triangle
	// worst case with restarts is 1 restart index and 3 indices per triangle
	(index_count / 3) * 5
}

/// Converts a triangle strip to a triangle list.
///
/// Returns the number of indices in the resulting list, with destination containing new index data.
///
/// # Arguments
///
/// * `destination`: must contain enough space for the target index buffer, worst case can be computed with [unstripify_bound]
pub fn unstripify(destination: &mut [u32], indices: &[u32], restart_index: u32) -> usize {
	let mut offset = 0;
	let mut start = 0;

	for (i, index) in indices.iter().enumerate() {
		if restart_index != 0 && *index == restart_index {
			start = i + 1;
		} else if i - start >= 2 {
            let mut a = indices[i - 2];
            let mut b = indices[i - 1];
            let c = indices[i - 0];

			// flip winding for odd triangles
			if ((i - start) & 1) != 0 {
				std::mem::swap(&mut a, &mut b);
			}

			// although we use restart indices, strip swaps still produce degenerate triangles, so skip them
			if a != b && a != c && b != c {
				destination[offset + 0] = a;
				destination[offset + 1] = b;
				destination[offset + 2] = c;
				offset += 3;
			}
		}
	}

	offset
}

pub fn unstripify_bound(index_count: usize) -> usize {
	assert!(index_count == 0 || index_count >= 3);

	if index_count == 0 { 0 } else { (index_count - 2) * 3 }
}
