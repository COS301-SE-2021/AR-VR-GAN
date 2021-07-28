using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public static class AreaPlane
{
    /**
     * Set the initial floor plane for a user to walk on
     * 10x10 square plane
     */
    public static GameObject CreateFloorPlane()
    {
        GameObject go = new GameObject("FloorPlane");
        MeshFilter mf = go.AddComponent(typeof(MeshFilter)) as MeshFilter;
        MeshRenderer mr = go.AddComponent(typeof(MeshRenderer)) as MeshRenderer;
        int size = 10;

        Mesh m = new Mesh();
        m.vertices = new Vector3[]
        {
            new Vector3(size, 0, size),
            new Vector3(size, 0, -size),
            new Vector3(-size, 0, size),
            new Vector3(-size, 0, -size)
        };

        m.uv = new Vector2[]
        {
            new Vector2(0, 0),
            new Vector2(0, 1),
            new Vector2(1, 1),
            new Vector2(1, 0)
        };

        m.triangles = new int[] {0, 2, 3, 0, 1, 3};
        
        if (mf != null)
            mf.mesh = m;
        m.RecalculateBounds();
        m.RecalculateNormals();

        return go;
    }
    
}
