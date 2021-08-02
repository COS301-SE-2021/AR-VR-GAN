using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR;

public class HMDInfoManager : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        if (!XRSettings.isDeviceActive)
        {
            print("NO HEADSET");
        }
        else if (XRSettings.isDeviceActive &&
                 (XRSettings.loadedDeviceName == "MOck HMD" || XRSettings.loadedDeviceName == "MockHMDDisplay"))
        {
            print("Using Mock");
        }
        else
        {
            print("Headset Available " + XRSettings.loadedDeviceName);
        }
        
    }
}
