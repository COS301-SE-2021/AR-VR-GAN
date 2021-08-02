using System;
using System.Collections;
using System.Collections.Generic;
using UnityEditor.ShaderGraph.Serialization;
using UnityEngine;
using UnityEngine.Networking;

public class TrackCoords : MonoBehaviour
{
    private void FixedUpdate()
    {
        Vector3 coords = transform.position;
        //print("x: " + coords.x + "\ny: " + coords.y + "\nz: " + coords.z);
        //print(transform.name);
    }

    private void Request()
    {
        List<MultipartFormDataSection> form = new List<MultipartFormDataSection>();
        form.Add(new MultipartFormDataSection("x"));
        UnityWebRequest request = new UnityWebRequest();
        request.SetRequestHeader("Content-Type", "application/json");
    }
}
