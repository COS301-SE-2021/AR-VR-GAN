using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MovementOfPicture : MonoBehaviour
{
    public Transform target;
    public Vector3 offset;
    public float smooth = 0.125f;
    // Update is called once per frame
    void FixedUpdate()
    {
        Vector3 pos = target.position + offset;
        Vector3 smoothPos = Vector3.Lerp(transform.position, pos, smooth);
        transform.position = smoothPos;
        
        transform.LookAt(target);
    }
}
