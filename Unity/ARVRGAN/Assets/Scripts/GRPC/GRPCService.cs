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

    private void Start()
    {
        plane.GetComponent<Renderer>().material.color = Color.white;
        pythonClient = new GenerativeModel();

    }

    // Start is called before the first frame update
    void LateUpdate()
    {
        if (camera.transform.position != pastCoords)
        {
            pythonClient.FetchImagePython(plane, camera);
        }

        pastCoords = camera.transform.position;
        //plane.GetComponent<Renderer>().material.color = Color.cyan;
        //client = new GRPCClient();
        //client.HandleCoords(plane, camera);
        //GenerativeModel gm = new GenerativeModel();
        //gm.FetchImagePython();
    } }
