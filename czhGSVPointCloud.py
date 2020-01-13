import numpy as np
import requests
import os
import math
import json
import random
import czhUtil
import grequests
from io import BytesIO
from PIL import Image
import numpy as np
import time
from scipy import interpolate


TILE_SIZE = 512
IMAGE_EXT =['.tif','.jpg','.png']
WORLD_EXT =['.tfw','jgw','pgw']



class czhGSVPano():
    #
    def __init__(self,lat='',lon='',zoom=4,style =1,panoid=''):
        self.pano_zoom = zoom
        self.pano_id= panoid

        if len(lat)>0:
            self.lat = float(lat.strip())
        if len(lon):
            self.lon =float(lon.strip())

        self.imageCounter =0
        self.totalImages =0
        #
        self.pano_location_prop = {}
        self.pano_proj_prop = {}
        self.links =[]
        self.panoTiles =[]

        if len(panoid)>0:
            exce_code =self.getPanoMetaData(2)
        else:
            exce_code = self.getPanoMetaData(1)
        if exce_code:
            if style ==1:
                self.getPanoTiles()
            else:
                self.getPanoImage(self.pano_id) #suffix='color'

    def getPanoTileXYfromURL(self,url):
        #return value 0
        xylist =["x=","y="]
        if len(str.strip(url))>0:
            splits_ = url.split("&")
            xy_idx = [i for i,x in enumerate(splits_) if any(thing in x for thing in xylist)]
            if len(xy_idx)==len(xylist):
                x = str.strip(splits_[xy_idx[0]].split("=")[1])
                y = str.strip(splits_[xy_idx[1]].split("=")[1])
                return {'exec_code':1,'XY':[x,y]}
        return {'exec_code':1,'XY':[]}

    def getPanoJson(self,style):
        if style ==1:
            url = 'http://maps.google.com/cbk?output=json&ll={},{}'.format(self.lat,self.lon)
        else:
            url = 'http://maps.google.com/cbk?output=json&panoid={}'.format(self.pano_id)

        try:
            r = requests.get(url)
            jdata = r.json()
            return jdata
        except Exception as e:
            print("Error in getPanoJson():", str(e))
            return None

    #get pano medat
    def getPanoMetaData(self,style=1):
        pano_meta_data = self.getPanoJson(style)

        if pano_meta_data != None:
            #get pano_location_prop
            self.pano_location_prop['panoId'] = pano_meta_data['Location']['panoId']
            if self.pano_id != self.pano_location_prop['panoId']:
                self.pano_id = self.pano_location_prop['panoId']

            # self.pano_location_prop['num_zoom_levels'] = pano_meta_data['Location']['num_zoom_levels']
            self.pano_location_prop['lat'] = float(pano_meta_data['Location']['lat'])
            self.pano_location_prop['lng'] = float(pano_meta_data['Location']['lng'])
            self.pano_location_prop['original_lat'] = float(pano_meta_data['Location']['original_lat'])
            self.pano_location_prop['original_lng'] = float(pano_meta_data['Location']['original_lng'])
            self.pano_location_prop['elevation_wgs84_m'] = float(pano_meta_data['Location']['elevation_wgs84_m'])
            self.pano_location_prop['elevation_egm96_m'] = float(pano_meta_data['Location']['elevation_egm96_m'])

            #get  projection info
            self.pano_proj_prop['pano_yaw_deg'] = float(pano_meta_data['Projection']['pano_yaw_deg'])
            self.pano_proj_prop['tilt_yaw_deg'] = float(pano_meta_data['Projection']['tilt_yaw_deg'])
            self.pano_proj_prop['tilt_pitch_deg'] =float(pano_meta_data['Projection']['tilt_pitch_deg'])

            #get link panos
            self.links = [link for link in pano_meta_data['Links']]

            return 1
        return 0


    def getPanoImage(self,PanoID,filepath='',prefix='',suffix='',ext=''):
        if len(filepath.strip())==0 :
            filepath = os.getcwd()+ r'K:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\Essex_test\jpg_pitch0\segmented_1024'
        if len(ext.strip())==0:
            ext = '.png'

        prefix = str(prefix)
        suffix = str(suffix)
        if prefix != "":
            prefix += '_'
        if suffix != "":
            suffix = '_' + suffix

        filename = filepath +prefix +self.pano_id+suffix+ext
        if os.path.exists(filename):
            self.panorama = Image.open(filename)

    #get panoramic image
    def getPanoTiles(self):
        #get panorama from internet

        w = 2**self.pano_zoom
        h = 2**(self.pano_zoom-1)
        self.totalImages = w*h
        self.imageCounter =0

        requests_list =[]
        for y in range(h):
            for x in range(w):
                num = random.randint(0, 3)
                # url = 'https://geo' + str(num) + '.ggpht.com/cbk?cb_client=maps_sv.tactile&authuser=0&hl=zh-CN&gl=us&panoid=' +self.pano_location_prop['panoId'] + \
                #       '&output=tile&x=' + str(x) + '&y=' + str(y) + '&zoom='+str(self.pano_zoom)+'&nbt&fover=2'

                url = "http://maps.google.com/cbk?output=tile&panoid="+self.pano_location_prop['panoId']+"&zoom=" + str(self.pano_zoom) + "&x=" + \
                      str(x) + "&y=" + str(y);
                resp = grequests.get(url,hooks={'response':self.urlResponse })
                requests_list.append(resp)

        grequests.map(requests_list,size=4)


    def urlResponse(self,response,**kwargs):

        if response.status_code ==200 :
            if "output=tile" in response.url:
                if self.getPanoTileXYfromURL(response.url)['exec_code']:
                    sX = self.getPanoTileXYfromURL(response.url)['XY'][0]
                    sY = self.getPanoTileXYfromURL(response.url)['XY'][1]
                    if len(sX)>0 and len(sY)>0:
                        pano_tile ={}

                        pano_tile['x'] = int(sX)
                        pano_tile['y'] = int(sY)
                        pano_tile['image'] = Image.open(BytesIO(response.content))

                self.panoTiles.append(pano_tile)

        self.imageCounter +=1
        if self.imageCounter == self.totalImages:
            self.constructPanoImage(1)

    def constructPanoImage(self,save_flag):
        # pano_w = 416*(2**self.pano_zoom)
        # pano_h = 416*(2**(self.pano_zoom-1))

        pano_w = 512*(2**self.pano_zoom)
        pano_h = 512*(2**(self.pano_zoom-1))


        self.panorama = Image.new('RGB', ( pano_w,  pano_h))

        for pano_tile in self.panoTiles:
            self.panorama.paste(pano_tile['image'],(TILE_SIZE*pano_tile['x'],TILE_SIZE*pano_tile['y'],
                                               TILE_SIZE*(pano_tile['x']+1),TILE_SIZE*(pano_tile['y']+1)))

        if save_flag:
            self.outputPanoImage(self.panorama)

    def outputPanoImage(self,img,saved_path="", prefix="", suffix=""):
        prefix = str(prefix)
        suffix = str(suffix)
        if prefix != "":
            prefix += '_'
        if suffix != "":
            suffix = '_' + suffix

        if len(saved_path)==0:
            saved_path = os.getcwd() +"\\Pano_Depthmap\\"

        if not os.path.exists(saved_path):
            os.makedirs(saved_path)

        file_name = os.path.join(saved_path, (prefix + self.pano_location_prop['panoId'] + '_' +
                                str(self.pano_location_prop['lat']) + '_' + str(self.pano_location_prop['lng']) + suffix + '.jpg'))
        img.save(file_name)

class czhGSVDepthmap():
    def __init__(self,lat ='',lon='',panoid='',save_depthmap_flag =1 ):
        if len(lat)>0:
            self.lat = float(lat.strip())
        if len(lon):
            self.lon =float(lon.strip())

        # self.lat = lat
        # self.lon = lon

        # self.json_data = None
        self.panoid = panoid
        self.depthmap_width = 512
        self.depthmap_height = 256
        self.save_depthmap_flag = save_depthmap_flag

        if len(panoid)>0:
            self.depthmap = self.decodeDepthmap(style = 2,save_flag=1)
        else:
            self.depthmap = self.decodeDepthmap(style=1,save_flag=1)


    def outputDepthmapJson(self,depthmap_json_data,saved_path='', prefix='', suffix=''):
        prefix = str(prefix)
        suffix = str(suffix)
        if prefix != "":
            prefix += '_'
        if suffix != "":
            suffix = '_' + suffix

        if len(saved_path)==0:
            saved_path = os.getcwd() +"\\Json\\"

        if not os.path.exists(saved_path):
            os.makedirs(saved_path)

        if depthmap_json_data != None:
            json_file_name = os.path.join(saved_path, prefix + depthmap_json_data['Location']['lng'] + '_' + depthmap_json_data['Location']['lat'] + '_' + depthmap_json_data['Location']['panoId'] + suffix + '.json')
            with open(json_file_name, 'w') as f:
                json.dump(depthmap_json_data, f)
        else:
            print("Fail to save depth map json for PanoID: " + depthmap_json_data['Location']['panoId'] + " !")
        pass

    def getDepthmapJson(self,style=1,dm=1,save_flag=0):
        #style =1 get json from longitude,latitude ;style =2 get json from panoid

        if style ==1:
            url = 'http://maps.google.com/cbk?output=json&ll={},{}&dm={}'.format(self.lat,self.lon,dm)
        else:
            url = 'http://maps.google.com/cbk?output=json&panoid={}&dm={}'.format(self.panoid,dm)

        try:
            r = requests.get(url)
            jdata = r.json()
            self.lat =  float(jdata['Location']['lat'])
            self.lon =  float(jdata['Location']['lng'])
            if self.panoid != jdata['Location']['panoId']:
                self.panoid = jdata['Location']['panoId']

            if save_flag == 1:
                self.outputDepthmapJson(jdata)
            return jdata
        except Exception as e:
            print("Error in getPanoIdDepthmapfrmLonlat():", str(e))
            return None

    def decodeDepthmap(self,style = 1,save_flag=0 ):
        #check use longitude,latitude or panoid to get depthmap json data
        self.json_data = self.getDepthmapJson(style,save_flag=save_flag)
        if self.json_data != None:
            self.panoid = self.json_data['Location']['panoId']
            depth_map_base64 = self.json_data['model']['depth_map']

            depthMap = np.array([d for d in czhUtil.parse(depth_map_base64)])
            headersize = depthMap[0]
            numberofplanes = czhUtil.getUInt16(depthMap,1)
            self.depthmap_width = czhUtil.getUInt16(depthMap,3)
            self.depthmap_height = czhUtil.getUInt16(depthMap,5)
            offset = czhUtil.getUInt16(depthMap,7)

            if (headersize != 8 or offset != 8):
                print("Unexpected depth map header! ")
                return None
            else:
                indices = []
                planes = []

                #depthMapIndices
                for i in range(self.depthmap_height * self.depthmap_width):
                    indices.append(depthMap[offset + i])

                #depthMapPlanes
                for i in range(numberofplanes):
                    byteOffset = offset + self.depthmap_width*self.depthmap_height + i * 4 * 4
                    n = [0, 0, 0]
                    n[0] = czhUtil.getFloat32(depthMap, byteOffset)
                    n[1] = czhUtil.getFloat32(depthMap, byteOffset + 4)
                    n[2] = czhUtil.getFloat32(depthMap, byteOffset + 8)
                    d = czhUtil.getFloat32(depthMap, byteOffset + 12)
                    planes.append({"n": n, "d": d})

                #construct depth map
                return  self.constructDepthMap(self.depthmap_width,self.depthmap_height,indices,planes)
        else:
            print("error in decodeDepthmap!")
            return None

    def outputSourceDepthmap2Image(self,depthmap_source_data,saved_path='', prefix='', suffix='',ext='' ):
        prefix = str(prefix)
        suffix = str(suffix)
        ext = str(ext)

        if prefix != "":
            prefix += '_'
        if suffix != "":
            suffix = '_' + suffix

        if len(ext) > 0:
            if ext not in  IMAGE_EXT:
                ext = '.tif'
        else:
            ext ='.tif'

        if len(saved_path)==0:
            saved_path = os.getcwd() +"\\Pano_Depthmap\\"

        if not os.path.exists(saved_path):
            os.makedirs(saved_path)

        # dp_duplicate = depthmap_source_data
        im = depthmap_source_data.reshape((self.depthmap_height,self.depthmap_width))

        file_name = os.path.join(saved_path, (prefix + self.json_data['Location']['panoId'] + '_' +
                                str(self.json_data['Location']['lat']) + '_' + str(self.json_data['Location']['lng']) + suffix + ext))

        img = Image.fromarray(im)
        img.save(file_name)

    def constructDepthMap(self,width,height,depthmapIndices,depthmapPlanes):

        v = [0, 0, 0]
        depthmap  = np.empty(width * height)
        #theta,phi
        sin_theta = np.empty(width)
        cos_theta = np.empty(width)
        sin_phi = np.empty(height)
        cos_phi = np.empty(height)

        for x in range(width):
            xnormalize = (width-x-1.0)/(width-1.0)
            theta = xnormalize* 2 * np.pi + np.pi / 2
            # theta = xnormalize * 2 * np.pi
            sin_theta[x] = np.sin(theta)
            cos_theta[x] =np.cos(theta)

        for y in range(height):
            ynormalize = (height-y-1.0)/(height-1.0)
            phi = ynormalize* np.pi
            sin_phi[y]= np.sin(phi)
            cos_phi[y] =np.cos(phi)

        for y in range(height):
            for x in range(width):
                v[0] = sin_phi[y]*cos_theta[x]
                v[1] = sin_phi[y] * sin_theta[x]
                v[2] = cos_phi[y]

                planeIdx = depthmapIndices[y*width +x]
                if planeIdx>0:
                    plane = depthmapPlanes[planeIdx]
                    t = np.abs(
                        plane["d"]
                        / (
                                v[0] * plane["n"][0]
                                + v[1] * plane["n"][1]
                                + v[2] * plane["n"][2]
                        )
                    )
                    depthmap[y*width+(width-x-1)]=t
                    # depthmap[y*width+x] =t
                else:
                    # depthmap[y*width+(width-x-1)] =0.0
                    depthmap[y * width + x] = 0.0
        if depthmap.size !=0:
            if self.save_depthmap_flag:
                self.outputSourceDepthmap2Image(depthmap)
            pass
        return depthmap

class czhGSVPointClouds():
    # rotate matrix refer to https: // en.wikipedia.org / wiki / Rotation_matrix
    #resolution unit :meter
    def __init__(self,panorama:czhGSVPano, depthmap:czhGSVDepthmap,save_xyz_flag:int =0,save_pointclouds2img_flag:int =1,resolution:float=0.1):

        self.c_depthmap = depthmap
        self.c_panorama = panorama
        self.resolution = resolution

        self.save_xyz_flag = save_xyz_flag
        self.save_pointclouds2img_flag = save_pointclouds2img_flag

        self.getPointsCloudWorldCoord()


    def pointCloud_to_image(self, rawPointsCloud, resolution):
        try:
            if type(rawPointsCloud) is np.ndarray:
                pointsCloud = rawPointsCloud
            else:
                pointsCloud = np.array(rawPointsCloud)

            minX =  min(pointsCloud[:, 0])
            maxY = max(pointsCloud[:, 1])
            rangeX = max(pointsCloud[:, 0]) - minX
            rangeY = maxY - min(pointsCloud[:, 1])
            w = int(rangeX / self.resolution)
            h = int(rangeY / self.resolution)

            _,pnt_pros = pointsCloud.shape
            n_colors = pnt_pros-3
            if n_colors ==1 :
                np_image = np.zeros(h*w*n_colors, dtype=np.uint8).reshape(h,w)
            else:
                np_image = np.zeros(h*w*n_colors, dtype=np.uint8).reshape(h,w,n_colors)
            #
            # print('rangeX, rangeY, w, h:', rangeX, rangeY, w, h)
            for point in pointsCloud:
                # print("point: ", point)
                col = int((point[0] - minX) / self.resolution)
                row = int((maxY - point[1]) / self.resolution)

                if row == h:
                    row = h - 1
                if col == w:
                    col = w - 1
                if n_colors ==1:
                    np_image[row][col] =point[3]
                else:
                    for i in range(n_colors):
                        np_image[row][col][i] = point[3+i]

            worldfile = [self.resolution, 0, 0, -self.resolution, minX, maxY]

            return np_image, worldfile

        except Exception as e:
            print("Error in pointCloud_to_image():", e)

    def rotate_x(self,pitch):
        #picth is degree
        r_x = np.array([[1.0,0.0,0.0],
                        [0.0,math.cos(pitch*math.pi/180.0),-1*math.sin(pitch*math.pi/180.0)],
                        [0.0,math.sin(pitch*math.pi/180.0),math.cos(pitch*math.pi/180.0)]])
        return r_x

    def rotate_y(self,yaw):
        #
        r_y = np.array([[math.cos(yaw*math.pi/180.0),0.0,math.sin(yaw*math.pi/180.0)],
                        [0.0,1.0,0.0],
                        [-1*math.sin(yaw*math.pi/180.0),0.0,math.cos(yaw*math.pi/180.0)]])
        return r_y

    def rotate_z(self,roll):
        #
        r_z = np.array([[math.cos(roll*math.pi/180.0),-1*math.sin(roll*math.pi/180.0),0.0],
                        [math.sin(roll*math.pi/180.0),math.cos(roll*math.pi/180.0),0.0],
                        [0.0,0.0,1.0]])
        return r_z

    def clip_pano0(self,theta0, phi0,tilt_pitch,tilt_yaw, fov_h, fov_v, width, img):
        """
        theta0 is  pitch
        phi0 is yaw
        render view at (pitch, yaw) with fov_h by fov_v
        width is the number of horizontal pixels in the view
        """
        m = self.rotate_y(phi0).dot(self.rotate_x(theta0)).dot(self.rotate_y(tilt_yaw)).dot(self.rotate_x(tilt_pitch))
        # m = np.dot(self.rotate_y(phi0), self.rotate_x(theta0))
        img = np.array(img)
        #   (base_height, base_width, _) = img.shape
        base_height = img.shape[0]
        base_width = img.shape[1]

        height = int(width * np.tan(fov_v / 2) / np.tan(fov_h / 2))

        new_img = np.zeros((height, width, 3), np.uint8)

        DI = np.ones((height * width, 3), np.int)
        trans = np.array([[2. * np.tan(fov_h / 2) / float(width), 0., -np.tan(fov_h / 2)],
                          [0., -2. * np.tan(fov_v / 2) / float(height), np.tan(fov_v / 2)]])

        xx, yy = np.meshgrid(np.arange(width), np.arange(height))

        DI[:, 0] = xx.reshape(height * width)
        DI[:, 1] = yy.reshape(height * width)

        v = np.ones((height * width, 3), np.float)

        v[:, :2] = np.dot(DI, trans.T)
        v = np.dot(v, m.T)

        diag = np.sqrt(v[:, 2] ** 2 + v[:, 0] ** 2)
        theta = np.pi / 2 - np.arctan2(v[:, 1], diag)
        phi = np.arctan2(v[:, 0], v[:, 2]) + np.pi

        ey = np.rint(theta * base_height / np.pi).astype(np.int)
        ex = np.rint(phi * base_width / (2 * np.pi)).astype(np.int)

        ex[ex >= base_width] = base_width - 1
        ey[ey >= base_height] = base_height - 1

        new_img[DI[:, 1], DI[:, 0]] = img[ey, ex]
        return new_img

    def clip_pano2(self, theta0, phi0,tilt_pitch,tilt_yaw, fov_h, fov_v, width, img):  # fov < 120
        """
          theta0 is pitch
          phi0 is yaw
          render view at (pitch, yaw) with fov_h by fov_v
          width is the number of horizontal pixels in the view
          """
        # m = np.dot(self.rotate_y(phi0), self.rotate_x(theta0))
        m = self.rotate_y(phi0).dot(self.rotate_x(theta0-tilt_pitch))
        img = np.array(img)
        try:
            (base_height, base_width, bands) = img.shape
        except:
            base_height,base_width = img.shape
            bands = 1
        # np.array(dm[dm['depthMap']]).reshape((dm["height"], dm["width"]))

        height =int(math.floor(width * np.tan(fov_v / 2) / np.tan(fov_h / 2)))
        width =int(width)

        if bands >1:
            new_img = np.zeros((height, width, bands), np.float)
            DI = np.ones((height * width, bands), np.int)
        else:
            # img = np.expand_dims(img, 3)
            # new_img = np.zeros((height, width,3), np.uint8)
            new_img= np.zeros((int(height),int( width)), np.float)
            DI = np.ones((int(height * width),3), np.int)

        # new_img = np.zeros((height, width, 3), np.uint8)
        # DI = np.ones((height * width, 3), np.int)
        trans = np.array([[2. * np.tan(fov_h / 2) / float(width), 0., -np.tan(fov_h / 2)],
                          [0., -2. * np.tan(fov_v / 2) / float(height), np.tan(fov_v / 2)]])

        xx, yy = np.meshgrid(np.arange(width), np.arange(height))

        DI[:, 0] = xx.reshape(height * width)
        DI[:, 1] = yy.reshape(height * width)

        v = np.ones((height * width, 3), np.float)

        v[:, :2] = np.dot(DI, trans.T)
        v = np.dot(v, m.T)

        diag = np.sqrt(v[:, 2] ** 2 + v[:, 0] ** 2)
        theta = np.pi / 2 - np.arctan2(v[:, 1], diag)
        phi = np.arctan2(v[:, 0], v[:, 2]) + np.pi

        ey = np.rint(theta * base_height / np.pi).astype(np.int)
        ex = np.rint(phi * base_width / (2 * np.pi)).astype(np.int)

        ex[ex >= base_width] = base_width - 1
        ey[ey >= base_height] = base_height - 1

        new_img[DI[:, 1], DI[:, 0]] = img[ey, ex]
        return new_img,theta.reshape(height,width),phi.reshape(height,width)

    def RawPointsCloud2Camera3D(self,pointsCloud,alpha,beta,gamma):
        #Euler angles:
        #here: alpha,beta,gamma correponding to pano_yaw_angle,tilt_pitch,tilt-yaw
        #
        #reference: https://en.wikipedia.org/wiki/Euler_angles

        #step 1 rotate alpha(pano_yaw_angle Ryzy
        #step 2 rotate beta
        #step 3 roate  gamma=0
        # gamma =0
        # beta=0

        # r_y = self.rotate_y(alpha)
        # # r_z = self.rotate_z(-beta)
        # r_x = self.rotate_x(-beta)
        # r_y1 = self.rotate_y(gamma)
        #
        # pointsCloud = np.matrix(r_y).dot(np.matrix(r_x)).dot(np.matrix(r_y1)).dot(pointsCloud.T).T

        # pointsCloud = np.matrix(self.rotate_y(-gamma)).dot(np.matrix(self.rotate_x(-beta))).dot(np.matrix(self.rotate_y(-alpha))).dot(pointsCloud.T).T

        # pointsCloud = np.matrix(self.rotate_z(-gamma)).dot(np.matrix(self.rotate_x(-beta))).dot(
        #     np.matrix(self.rotate_z(-alpha))).dot(pointsCloud.T).T

        # pointsCloud = self.rotate_z(-alpha).dot(self.rotate_x(-beta)).dot(self.rotate_z(-gamma)).dot(pointsCloud.T).T
        pointsCloud = self.rotate_z(-gamma).dot(self.rotate_x(-beta)).dot(self.rotate_z(-alpha)).dot(pointsCloud.T).T
        # pointsCloud = self.rotate_y(-alpha).dot(self.rotate_x(-beta)).dot(self.rotate_z(-gamma)).dot(pointsCloud.T).T
        return pointsCloud

    def PointsCloudCamera3D2World3D(self,pointsCloud):
        #x1,y1,z1
        #... ...    pointsCloud format :np.array
        #xn,yn,zn
        # step 1 :rotate z axis around x axis 90
        # r_x = self.rotate_x(90)

        #step 2 :rotate x axis around z axis 90
        # r_z = self.rotate_z(-90)

        # pointsCloud = r_z.dot(r_x).dot(pointsCloud.T).T
        # pointsCloud =np.matrix(self.rotate_z(-90)).dot(np.matrix(self.rotate_x(90))).dot(pointsCloud.T).T

        # pointsCloud = np.matrix(self.rotate_z(90)).dot(pointsCloud.T).T
        # pointsCloud = self.rotate_z(-90).dot(pointsCloud.T).T
        return pointsCloud

    def pointCloud2Image(self,rawPointsCloud,resolution = 0.25,saved_path='', prefix='', suffix='',ext=''):
        prefix = str(prefix)
        suffix = str(suffix)
        ext = str(ext)

        if prefix != "":
            prefix += '_'
        if suffix != "":
            suffix = '_' + suffix

        if len(ext) > 0:
            if ext not in  IMAGE_EXT:
                ext = '.tif'
                w_ext = '.tfw'
            else:
                w_ext = WORLD_EXT[IMAGE_EXT.index(ext)]
        else:
            ext ='.tif'
            w_ext ='.tfw'

        if len(saved_path)==0:
            saved_path = os.getcwd() +"\\Pano_Depthmap\\"

        if not os.path.exists(saved_path):
            os.makedirs(saved_path)

        #filter distance

        imgfile_name = os.path.join(saved_path, (prefix + self.c_panorama.pano_location_prop['panoId'] + '_' +
                    str(self.c_panorama.pano_location_prop['lat']) + '_' + str(self.c_panorama.pano_location_prop['lng']) + suffix + ext))
        #save image
        np_image, worldfile = self.pointCloud_to_image(rawPointsCloud,resolution)
        colored = czhUtil.get_color_pallete(np_image, 'ade20k')
        colored.save(imgfile_name)

        # img = Image.fromarray(np_image)
        # img.save(imgfile_name)

        #save image world file
        worldfile_name = os.path.splitext(imgfile_name)[0] +w_ext
        with open(worldfile_name,'wt') as wFile:
            if len(worldfile)==6 :
                wFile.write("%0.5f\n" % float(worldfile[0]))
                wFile.write("%0.5f\n" % float(worldfile[1]))
                wFile.write("%0.5f\n" % float(worldfile[2]))
                wFile.write("%0.5f\n" % float(worldfile[3]))
                wFile.write("%0.5f\n" % float(worldfile[4]))
                wFile.write("%0.5f\n" % float(worldfile[5]))
            wFile.close()

    # def getClipSection(self,theta,phi,fov_v,w,h):
    #     fov_h = math.atan((h * math.tan((math.radians(fov_v) / 2)) / w)) * 2
    #     fov_h = math.degrees(fov_h)
    #
    #     min_h_angle,max_h_angle = phi-fov_h/2.0,phi+fov_h/2.0
    #     min_v_angle,max_v_angle = theta-fov_v/2.0,theta+fov_v/2.0
    #
    #     return min_h_angle,min_v_angle,max_h_angle,max_v_angle
    # def constructRawPointsCloud2(self, fov_h, theta0, phi0, w_thumb, h_thumb):
    #     assert (self.c_panorama.pano_location_prop['panoId'] == self.c_depthmap.panoid)
    #     raw_pointsCloud = []
    #
    #     fov_v = math.atan((h_thumb * math.tan((math.radians(fov_h) / 2)) / w_thumb)) * 2
    #     fov_v = math.degrees(fov_v)
    #
    #     theta = theta0  # -self.c_panorama.pano_proj_prop['tilt_pitch_deg']
    #     phi = phi0 - self.c_panorama.pano_proj_prop['pano_yaw_deg']
    #     #
    #     # -self.c_panorama.pano_proj_prop['tilt_pitch_deg']-self.c_panorama.pano_proj_prop['tilt_yaw_deg']
    #     ndepthmap_data = np.array(self.c_depthmap.depthmap).reshape(self.c_depthmap.depthmap_height,
    #                                                                 self.c_depthmap.depthmap_height)
    #
    #     nImage = self.clip_pano2(theta, phi, 0, 0, math.radians(fov_h), math.radians(fov_v), w_thumb,
    #                              self.c_panorama.panorama)
    #     ndepthmap = self.clip_pano2(theta, phi, theta, phi, 0, 0, math.radians(fov_h), math.radians(fov_v), w_thumb, )
    #
    #     imgfile_name = r'D:\2019\njit learning\201909\streetView\StreetView\Pano_Depthmap\test.png'
    #     img = Image.fromarray(nImage)
    #     # w_thumb,h_thumb = img.size
    #     # img.save(imgfile_name)
    #     colored = czhUtil.get_color_pallete(nImage, 'ade20k')
    #     colored.save(imgfile_name)
    #
    #     #
    #     grid_col = np.linspace(-math.pi, math.pi, self.c_depthmap.depthmap_width)
    #     grid_row = np.linspace(math.pi / 2, -math.pi / 2, self.c_depthmap.depthmap_height)
    #
    #     depthmap_data = np.array(self.c_depthmap.depthmap).reshape(self.c_depthmap.depthmap_height,
    #                                                                self.c_depthmap.depthmap_width)
    #     interp = interpolate.interp2d(grid_col, grid_row, depthmap_data, kind='linear')
    #
    #     try:
    #         count = 0
    #         start_time = time.time()
    #         for y in range(h_thumb):
    #             # lat = ((h_thumb-y-1)*180/h_thumb -90)*math.pi / 180
    #             # lat = (h_thumb-y-1)*fov_v/(h_thumb-1.0)- fov_v/2.0
    #             lat = (h_thumb - y - 1.0) * fov_v / (h_thumb - 1.0) - fov_v / 2.0
    #             # lat1 = (h_thumb-y-1)*fov_v/(h_thumb-1.0)- fov_v/2.0+theta
    #             # if lat1>90:
    #             #     lat1 = lat1 - 180
    #             # elif lat1<-90:
    #             #     lat1 = lat1 + 180
    #             lat = math.radians(lat)
    #             r = math.cos(lat)
    #             for x in range(w_thumb):
    #                 # lng = ((1.0-x/w_thumb)*360-180)*math.pi / 180
    #                 # lng0 = - x / (w_thumb-1.0) * fov_h - fov_h/2.0+phi0
    #                 # lng = - x / (w_thumb-1.0) * fov_h - fov_h/2.0
    #                 # lng1 = - x / (w_thumb-1.0) * fov_h - fov_h/2.0+phi0
    #                 lng = (x / (w_thumb - 1.0)) * fov_h - fov_h / 2.0
    #                 lng1 = lng + phi0
    #
    #                 if lng1 > 180:
    #                     lng1 = lng1 - 360
    #                 elif lng1 < -180:
    #                     lng1 = lng1 + 360
    #
    #                 depth = interp(lng1, lat)[0]
    #                 lng = math.radians(lng)
    #                 # color_value = self.c_panorama.panorama.getpixel((x, y))
    #                 color_value = img.getpixel((x, y))
    #                 if depth > 0.0005 and depth < 20.0:  # and color_value ==6
    #
    #                     # pnt_x = r*math.cos(lng)*depth
    #                     # pnt_y = r*math.sin(lng)*depth
    #                     pnt_x = r * math.sin(lng) * depth
    #                     pnt_y = r * math.cos(lng) * depth
    #                     pnt_z = math.sin(lat) * depth
    #
    #                     if type(color_value) is tuple:
    #                         color_value = list(color_value)
    #                     else:
    #                         color_value = [color_value]
    #                     raw_pointsCloud.append([pnt_x, pnt_y, pnt_z] + list(color_value))
    #
    #                 if count % 100000 == 0:
    #                     t_100000 = time.time()
    #                     dt = t_100000 - start_time
    #
    #                     print("slapse:" + str(dt))
    #                 count = count + 1
    #     except Exception as e:
    #         print("Error in constructRawPointsCloud():", e)
    #
    #     return raw_pointsCloud  # ,pointsColor

    def constructRawPointsCloud2(self,fov_h,theta0,phi0,w_thumb,h_thumb):
        assert (self.c_panorama.pano_location_prop['panoId'] == self.c_depthmap.panoid)
        raw_pointsCloud = []

        fov_v = math.atan((h_thumb * math.tan((math.radians(fov_h) / 2)) / w_thumb)) * 2
        fov_v = math.degrees(fov_v)

        depth_height =self.c_depthmap.depthmap_height
        depth_width = self.c_depthmap.depthmap_width
        pano_width,pano_height = self.c_panorama.panorama.size


        theta = theta0  #-self.c_panorama.pano_proj_prop['tilt_pitch_deg']
        phi = phi0-self.c_panorama.pano_proj_prop['pano_yaw_deg']
        #
        #-self.c_panorama.pano_proj_prop['tilt_pitch_deg']-self.c_panorama.pano_proj_prop['tilt_yaw_deg']
        ndepthmap = np.array(self.c_depthmap.depthmap).reshape(depth_height, depth_width)

        # nImage = self.clip_pano2(theta, phi, 0, 0, math.radians(fov_h), math.radians(fov_v),w_thumb, self.c_panorama.panorama)

        nImage,pano_thetas,pano_phis = self.clip_pano2(theta,phi,0,0,math.radians(fov_h),math.radians(fov_v),w_thumb,self.c_panorama.panorama)
       # nImage, pano_thetas, pano_phis = self.clip_pano2(theta0, phi0, 0, 0, math.radians(fov_h), math.radians(fov_v), \
                                                   #      w_thumb, img_thumb)
        img = Image.fromarray(nImage)

        h_thumb,w_thumb = pano_phis.shape
        ndepthmap_clip = self.clip_pano2(theta,phi,0,0,math.radians(fov_h),math.radians(fov_v),depth_width*w_thumb/pano_width,ndepthmap.tolist())
        ndepthmap_clip= np.array(ndepthmap_clip)

        imgfile_name =r'D:\2019\njit learning\201909\streetView\StreetView\Pano_Depthmap\test1.png'
        # img.save(imgfile_name)
        colored = czhUtil.get_color_pallete(nImage, 'ade20k')
        colored.save(imgfile_name)

        #
        # grid_col = np.linspace(math.radians(-fov_h/2), math.radians(fov_h/2), ndepthmap_clip.shape[1])
        # grid_row = np.linspace(math.radians(fov_v/2), math.radians(-fov_v/2), ndepthmap_clip.shape[0])

        grid_col = np.linspace(-math.pi, math.pi, self.c_depthmap.depthmap_width)
        grid_row = np.linspace(math.pi, 0, self.c_depthmap.depthmap_height)

        depthmap_data = np.array(self.c_depthmap.depthmap).reshape(self.c_depthmap.depthmap_height,self.c_depthmap.depthmap_width)
        interp = interpolate.interp2d(grid_col, grid_row, depthmap_data, kind='linear')

        # interp = interpolate.interp2d(grid_col, grid_row, ndepthmap_clip, kind='linear')
        # dis=[]
        # h1  = grid_row.size
        # w1= grid_col.size
        # for i in range(h1):
        #     for j in range(w1):
        #       dis1 = interp(grid_col[j],grid_row[i])[0]
        #       dis.append(dis1)
        # dis = np.array(dis).reshape(h1,w1)



        try:
            count =0
            k=0
            start_time = time.time()
            for y in range(h_thumb):
                # lat = ((h_thumb-y-1)*180/h_thumb -90)*math.pi / 180
                # lat = (h_thumb - y - 1.0) *fov_v / (h_thumb-1.0) - fov_v/2.0
                # lat1 = (h_thumb-y-1)*fov_v/(h_thumb-1.0)- fov_v/2.0+theta
                # if lat1>90:
                #     lat1 = lat1 - 180
                # elif lat1<-90:
                #     lat1 = lat1 + 180
                # lat = math.radians(lat)
                # r = math.cos(lat)
                k =k+1
                for x in range(w_thumb):
                    # lng = ((1.0-x/w_thumb)*360-180)*math.pi / 180
                    # lng0 = - x / (w_thumb-1.0) * fov_h - fov_h/2.0+phi0
                    # lng = - x / (w_thumb-1.0) * fov_h - fov_h/2.0
                    # lng1 = - x / (w_thumb-1.0) * fov_h - fov_h/2.0+phi0
                    # lng = (x / (w_thumb-1.0)) * fov_h - fov_h/2.0
                    # lng1 = lng+ phi
                    #
                    # if lng1>180:
                    #     lng1 = lng1-360
                    # elif lng1<-180:
                    #     lng1 = lng1+360
                    #
                    # lng = math.radians(lng)
                    # depth = interp(lng,lat)[0]
                    # lng1 = math.radians(lng1)
                    #

                    # depth = interp1(lng1,lat)[0]
                    # print(depth,depth1)

                    lat1 = pano_thetas[y,x]
                    lat = lat1-math.pi/2

                    lng1 = pano_phis[y, x] - math.pi
                    lng = lng1-math.radians(phi)

                    if lng>math.pi:
                        lng = lng-math.pi*2.0
                    elif lng<-math.pi:
                        lng = lng+math.pi*2.0

                    depth = interp(lng1,lat1)[0]
                    r = math.cos(lat)
                    # color_value = self.c_panorama.panorama.getpixel((x, y))
                    color_value = img.getpixel((x,h_thumb-y-1))
                    if depth>0.0005 and depth<20.0 :#and color_value ==6

                        # pnt_x = r*math.cos(lng)*depth
                        # pnt_y = r*math.sin(lng)*depth
                        pnt_x = r*math.sin(lng)*depth
                        pnt_y = r*math.cos(lng)*depth
                        pnt_z = math.sin(lat)*depth

                        if type(color_value) is tuple:
                            color_value = list(color_value)
                        else:
                            color_value = [color_value]
                        raw_pointsCloud.append([pnt_x, pnt_y, pnt_z] + list(color_value))

                    if count%100000==0:
                        t_100000 = time.time()
                        dt = t_100000-start_time

                        print ("slapse:"+str(dt))
                    count=count+1
        except Exception as e:
            print("Error in constructRawPointsCloud2():", e)

        return raw_pointsCloud  #,pointsColor

    def constructRawPointsCloud(self):
        assert(self.c_panorama.pano_location_prop['panoId']==self.c_depthmap.panoid)

        depthmap_w,depthmap_h = self.c_depthmap.depthmap_width,self.c_depthmap.depthmap_height
        # panoimage_w,panoimage_h = self.c_panorama.panorama.size

        # self.c_panorama.panorama.thumbnail((depthmap_w,depthmap_h))
        w_thumb,h_thumb = self.c_panorama.panorama.size
        #create points cloud from
        raw_pointsCloud =[]
        # pointsColor =[]

        #get

        #interpolate
        # grid_col = np.linspace(math.pi, -math.pi, self.c_depthmap.depthmap_width)
        # grid_row = np.linspace(-math.pi / 2, math.pi / 2, self.c_depthmap.depthmap_height)

        #define
        grid_col = np.linspace(-math.pi, math.pi, depthmap_w)
        grid_row = np.linspace(math.pi / 2, -math.pi / 2,depthmap_h)

        depthmap_data = np.array(self.c_depthmap.depthmap).reshape(depthmap_h,depthmap_w)
        interp = interpolate.interp2d(grid_col, grid_row, depthmap_data, kind='linear')

        try:
            count =0
            start_time = time.time()
            # for x in range(depthmap_h):
            for y in range(h_thumb):
                # lat = ((h_thumb-y-1.0)*180/(h_thumb-1.0) -90)*math.pi / 180
                lat = (h_thumb-y-1.0)*180/(h_thumb-1.0) -90
                lat = math.radians(lat)
                r = math.cos(lat)
                for x in range(w_thumb):
                    # lng = ((x/(w_thumb-1.0))*360-180)*math.pi / 180
                    lng = (x/(w_thumb-1.0))*360-180
                    lng = math.radians(lng)
                    # depth = self.c_depthmap.depthmap[y*depthmap_w+(depthmap_h-x )]
                    depth = interp(lng,lat)[0]
                    if depth>0.0005 and depth<20.0 :#and color_value ==6
                        color_value = self.c_panorama.panorama.getpixel((x, y))
                        # pnt_x = -1.0 * r * math.cos(lng ) * depth
                        # pnt_y = math.sin(lat ) * depth
                        # pnt_z = r * math.sin(lng ) * depth

                        pnt_x = r*math.sin(lng)*depth
                        pnt_y = r*math.cos(lng)*depth
                        pnt_z = math.sin(lat)*depth

                        if type(color_value) is tuple:
                            color_value = list(color_value)
                        else:
                            color_value = [color_value]
                        raw_pointsCloud.append([pnt_x, pnt_y, pnt_z] + list(color_value))

                    if count%100000==0:
                        t_100000 = time.time()
                        dt = t_100000-start_time

                        print ("slapse:"+str(dt))
                    count=count+1

        except Exception as e:
            print("Error in constructRawPointsCloud():", e)
        #get pano_color
        # for y in range(depthmap_h):
        #     lat = y*180/depthmap_h -90
        #     r = math.cos(lat)
        #     for x in range(depthmap_w):
        #         lng = (1.0-x/depthmap_w)*360-180
        #         depth = self.c_depthmap.depthmap[y*depthmap_w+(depthmap_h-x )]
        #         if depth>0.005 and depth<12.0 :
        #
        #             # x_norm = (x)/depthmap_w
        #             # y_norm = y/depthmap_h
        #
        #             # x_color = int(x_norm*panoimage_w)-1
        #             # if x_color == panoimage_w:
        #             #     x_color =x_color-1
        #             # y_color = int(y_norm*panoimage_h)
        #             # if y_color == panoimage_h:
        #             #     y_color = y_color-1
        #
        #             # pointsColor.append([self.c_panorama.panorama.getpixel((x_color,y_color))])
        #
        #             pnt_x = -1.0*r*math.cos(lng*math.pi/180)*depth
        #             pnt_y = math.sin(lat*math.pi/180)*depth
        #             pnt_z = r*math.sin(lng*math.pi/180)*depth
        #
        #             color_value =self.c_panorama.panorama.getpixel((x,y))
        #             if type(color_value) is tuple:
        #                 color_value = list(color_value)
        #             else:
        #                 color_value =[color_value]
        #             raw_pointsCloud.append([pnt_x,pnt_y,pnt_z]+color_value)
        #             # raw_pointsCloud.append([pnt_x,pnt_y,pnt_z]+list(self.c_panorama.panorama.getpixel((x_color,y_color))))

        #ouput
        # if self.save_pointclouds2img_flag :
        #     self.pointCloud2Image(raw_pointsCloud,suffix='PtsCloud')

        return raw_pointsCloud  #,pointsColor

    def getPointsCloudWorldCoord(self):
        # rawPointsCloud x1,y1,z1
        #                ... ...  array list
        #                xn,yn,zn

        pano_id = self.c_panorama.pano_id
        pano_yaw = self.c_panorama.pano_proj_prop["pano_yaw_deg"]
        tilt_yaw = self.c_panorama.pano_proj_prop["tilt_yaw_deg"]
        tilt_pitch= self.c_panorama.pano_proj_prop["tilt_pitch_deg"]

        cam_pos_lat = self.c_panorama.pano_location_prop["lat"]
        cam_pos_lng = self.c_panorama.pano_location_prop["lng"]
        cam_pos_elev = self.c_panorama.pano_location_prop["elevation_egm96_m"]

        world_x,world_y =czhUtil.lonlat2WebMercator(cam_pos_lng,cam_pos_lat)

        # pointsCloud = np.array(self.constructRawPointsCloud())
        fov_h =90
        theta=0
        phi =0
        h_thumb =768
        w_thumb =1024

        pointsCloud = np.array(self.constructRawPointsCloud2(fov_h,theta,phi,w_thumb,h_thumb))

        pointsCloud_xyz = pointsCloud[:,:3]

        # pointsCloud_xyz[:] = self.RawPointsCloud2Camera3D(pointsCloud_xyz,pano_yaw,tilt_pitch,tilt_yaw)
        # pointsCloud_xyz[:] = self.RawPointsCloud2Camera3D(pointsCloud_xyz, pano_yaw, 0, 0)
        pointsCloud_xyz[:] = self.RawPointsCloud2Camera3D(pointsCloud_xyz, phi, theta, 0)
        pointsCloud_xyz[:] = self.PointsCloudCamera3D2World3D(pointsCloud_xyz)

        pointsCloud_xyz[:] = pointsCloud_xyz + np.array([world_x,world_y,cam_pos_elev])
        # pointsCloud_xyz[:] = pointsCloud_xyz

        if self.save_xyz_flag:
            resultDirctory = os.getcwd()+"\\result\\"
            if not os.path.exists(resultDirctory):
                os.makedirs(resultDirctory)

            str_xyzFile = resultDirctory + str(cam_pos_lat)+"_" +str(cam_pos_lng)+"_"+pano_id+".xyz"
            with open(str_xyzFile, 'wt') as xyzFile:
                if pointsCloud.size != 0:
                    rows,cols = pointsCloud.shape
                    # assert(cols ==3)
                    for row in range(rows):
                        # x = pointsCloud[row,0]
                        # y = pointsCloud[row,1]
                        # z = pointsCloud[row,2]
                        # xyzFile.write("%0.2f,%0.2f,%0.3f\n"%(x,y,z))
                        sLine =','.join('%0.3f'%e for e in pointsCloud[row])
                        xyzFile.write(sLine +'\n')
                xyzFile.close()

        if self.save_pointclouds2img_flag:
            self.pointCloud2Image(pointsCloud,resolution=0.25,suffix='PtsCloud')

        return pointsCloud

class czhGSVPanoTree():
    def __init__(self):
        pass


#test
import multiprocessing as mp

def GSVPointClouds_Generate_mp(panoids):
    # Step 4: Close Pool and let all the processes complete
    Process_cnt =mp.cpu_count()
    pool = mp.Pool(processes=Process_cnt)
    for i in range(Process_cnt):
        pool.apply_async(GsvPointClouds_Generate, args=(panoids))
    pool.close()
    pool.join()

def GsvPointClouds_Generate(panoids):
    cur_count =0
    for panoid in panoids:
        t_depthmap = czhGSVDepthmap(panoid=panoid)
        t_pano = czhGSVPano(panoid=panoid, style=2)

        t_pointClouds = czhGSVPointClouds(t_pano, t_depthmap)

        print(str(cur_count),panoid)

def main():

    # Dg0LGX581Uu5ooKrLlvQGA f9IgU8o1UG5KztkMJIYCJg qAYd9QiwszKOAKE7TGmfuQ kE3e_0Ylr8kfcOkDJHXb1w ，OnscXILHuyqAJrWW8Bx2AQ   jS7rN0Wca908ifzbSUfi7Q  Washington St 4oHz09u-8JQ8aFfAcQKuRg iQ9Yk6re834ZluvfQXkt5g
    t_depthmap = czhGSVDepthmap(panoid='Dg0LGX581Uu5ooKrLlvQGA') #5ywu6UsJ7BjbquAj8yzkIw

    # t_depthmap = czhGSVDepthmap(panoid='iMYzd7bDv0J0CKfdLzkAZg')
    t_pano= czhGSVPano(panoid='Dg0LGX581Uu5ooKrLlvQGA',style =2)

    file = r'K:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\Essex_test\jpg_pitch0\segmented_1024\color\ilnBcF0ZOVDY-VIjgYTiJg_-74.207665_40.794218_0_184.00_color.png'

    t_depthmap = czhGSVDepthmap(panoid='Dg0LGX581Uu5ooKrLlvQGA') #5ywu6UsJ7BjbquAj8yzkIw
    t_pano= czhGSVPano(panoid='Dg0LGX581Uu5ooKrLlvQGA',style =2)

    t_pointClouds = czhGSVPointClouds(t_pano,t_depthmap)

    print(t_pointClouds)

    # path = r"D:\2019\njit learning\201909\streetView\StreetView\Pano_Segment"
    # panoids = []
    #
    # czhUtil.getfilenamefromfilepath(path, panoids)
    #
    # GSVPointClouds_Generate_mp(panoids)

    # panoids = reversed(panoids)

    # cur_count =0
    # for panoid in panoids:
    #     if(cur_count>140):
    #         t_depthmap = czhGSVDepthmap(panoid=panoid)  # 5ywu6UsJ7BjbquAj8yzkIw
    #         t_pano = czhGSVPano(panoid=panoid, style=2)
    #
    #         t_pointClouds = czhGSVPointClouds(t_pano, t_depthmap)
    #
    #
    #         print(str(cur_count),panoid)
    #     # print(panoid)
    #
    #     cur_count = cur_count+1

if __name__ == "__main__":
    main()
