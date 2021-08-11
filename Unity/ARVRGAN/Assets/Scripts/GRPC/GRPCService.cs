using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class GRPCService : MonoBehaviour
{
    //private Image;
    private GRPCClient client;

    private GenerativeModel pythonClient;
    //public Material imgPlane;
    public GameObject plane;
    public GameObject camera;
    public Vector3 pastCoords;
    private int interval = 100;

    private void Start()
    {
        plane.GetComponent<Renderer>().material.color = Color.white;
        pythonClient = new GenerativeModel();

    }

    // Start is called before the first frame update
    void LateUpdate()
    {
        if (Time.frameCount % interval == 0)
        {
            Vector3 current = camera.transform.position;
            //double sum = (current.x+5)/10 + (current.y+5)/10 + (current.z+5)/10;
            //double pastSum = (pastCoords.x+5)/10 + (pastCoords.y+5)/10 + (pastCoords.z+5)/10;
            //if ((sum > (pastSum + 0.2)) ||  (sum < (pastSum + 0.2)))
            //{
                print("fetching new image");
                pythonClient.FetchImagePython(plane, camera);
            //}

            pastCoords = current;
        }

        //plane.GetComponent<Renderer>().material.color = Color.cyan;
        //client = new GRPCClient();
        //client.HandleCoords(plane, camera);
        //GenerativeModel gm = new GenerativeModel();
        //gm.FetchImagePython();
    } }
