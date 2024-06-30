import { useRef } from 'react'
import './App.css'

const FileUpload = () => {
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
        console.log(typeof(e))
        const files = e.target.files;
        console.log(files)
    };

    const handleClick = () => {
        if (fileInputRef.current) {
            fileInputRef.current.click();
        }
    };

    return (
        <div className="flex items-center justify-center mt-8">
            <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileUpload}
                accept="image/*"
                multiple
                className="hidden"
            />
            <button
                className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline"
                onClick={handleClick}
            >
                Select Image
            </button>
        </div>
    );
};


function App() {
    return (
        <div>
            <FileUpload />
        </div>
    )
}

export default App
