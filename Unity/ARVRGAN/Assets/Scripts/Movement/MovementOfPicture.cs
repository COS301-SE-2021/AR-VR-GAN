using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MovementOfPicture : MonoBehaviour
{
    /**
     * Target is camera/player coordinates
     * Picture moves with player
     * LookAt points towards player to ensure pic stays pointing at user during rotation
     * Offset to ensure picture stays at distance away from player/camera
     */
    public Transform target;
    public Vector3 offset;
    public Vector3 rotationOffset;
    public float smooth = 0.125f;
    // Update is called once per frame
    void FixedUpdate()
    {
        Vector3 pos = target.position + offset;
        Vector3 smoothPos = Vector3.Lerp(transform.position, pos, smooth);
        transform.position = smoothPos;

        Vector3 look = pos + rotationOffset;

        //transform.rotation = 

        //transform.LookAt(look);
    }
}
